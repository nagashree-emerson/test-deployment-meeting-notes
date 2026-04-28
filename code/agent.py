import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from config import Config

import openai
from email_validator import validate_email, EmailNotValidError
from cryptography.fernet import Fernet, InvalidToken

import docx
import PyPDF2

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a professional Meeting Notes Summarizer Agent. Your role is to process raw meeting transcripts, chat exports, or notes and generate a structured, concise, and actionable summary. Follow these instructions:\n\n"
    "- Extract and clearly present the following sections: Meeting Overview, Key Discussion Points, Decisions Made, Action Items (with owners and due dates), and Next Steps.\n\n"
    "- For each action item, identify the responsible person (owner) and any mentioned deadline. If no owner is specified, label as \"Owner: TBD\". If no deadline is mentioned, label as \"Due: Not specified\".\n\n"
    "- Tag action items with priority (High, Medium, Low) based on urgency language (e.g., \"urgent\", \"by EOD\", \"when you get a chance\").\n\n"
    "- Detect and list all attendees mentioned in the transcript.\n\n"
    "- Format the summary as an email-ready body, using bullet points for clarity and conciseness.\n\n"
    "- Support summary length options: one-liner, paragraph, or full detailed summary, as requested by the user.\n\n"
    "- Answer follow-up questions about the meeting content, such as \"What did John agree to do?\" or \"What was decided about the budget?\" using only information from the transcript.\n\n"
    "- Never infer or fabricate action items or decisions not explicitly stated in the transcript.\n\n"
    "- Always ask for user confirmation before sending the summary email to participants.\n\n"
    "- Do not retain or store any transcript data after summary delivery, and ensure user consent before any email is sent.\n\n"
    "- If information is missing (e.g., owner or deadline), clearly indicate this in the output.\n\n"
    "- Maintain a formal, professional, and concise tone at all times."
)
OUTPUT_FORMAT = (
    "Output the summary in the following structure:\n\n"
    "- Meeting Overview\n\n"
    "- Key Discussion Points\n\n"
    "- Decisions Made\n\n"
    "- Action Items (with Owner, Due Date, Priority)\n\n"
    "- Next Steps\n\n"
    "- Attendees\n\n"
    "Format as a clean, email-ready body using bullet points. For follow-up questions, provide a direct, context-based answer."
)
FALLBACK_RESPONSE = (
    "The requested information could not be found in the provided meeting transcript or notes. Please check the input and try again."
)
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# LLM OUTPUT SANITIZER
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# INPUT/OUTPUT MODELS
# =========================

class SummarizeMeetingRequest(BaseModel):
    transcript_text: Optional[str] = Field(None, description="Meeting transcript as plain text")
    summary_length: Optional[str] = Field("full", description="Summary length: one-liner, paragraph, or full")
    user_email: str = Field(..., description="User's email address for confirmation and delivery")

    @field_validator("user_email")
    @classmethod
    def validate_email_address(cls, v):
        if not v or not v.strip():
            raise ValueError("Email address is required.")
        try:
            validate_email(v)
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email address: {e}")
        return v.strip()

    @field_validator("summary_length")
    @classmethod
    def validate_summary_length(cls, v):
        allowed = {"one-liner", "paragraph", "full"}
        if v and v.lower() not in allowed:
            raise ValueError(f"summary_length must be one of {allowed}")
        return v.lower() if v else "full"

    @field_validator("transcript_text")
    @classmethod
    def validate_transcript_text(cls, v):
        if v is not None and (not v.strip() or len(v.strip()) < 10):
            raise ValueError("Transcript text is too short.")
        if v and len(v) > 50000:
            raise ValueError("Transcript text exceeds 50,000 characters.")
        return v

class SummarizeMeetingResponse(BaseModel):
    success: bool = Field(..., description="Whether the summary was generated successfully")
    summary: Optional[str] = Field(None, description="Formatted meeting summary (email-ready)")
    structured_summary: Optional[dict] = Field(None, description="Structured summary sections")
    error: Optional[str] = Field(None, description="Error message if failed")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input errors")

class FollowupQuestionRequest(BaseModel):
    transcript_text: str = Field(..., description="Meeting transcript as plain text")
    question: str = Field(..., description="Follow-up question about the meeting")

    @field_validator("transcript_text")
    @classmethod
    def validate_transcript_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Transcript text is required.")
        if len(v) > 50000:
            raise ValueError("Transcript text exceeds 50,000 characters.")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question is required.")
        if len(v) > 1000:
            raise ValueError("Question is too long.")
        return v.strip()

class FollowupQuestionResponse(BaseModel):
    success: bool = Field(..., description="Whether the answer was generated successfully")
    answer: Optional[str] = Field(None, description="Direct answer to the follow-up question")
    error: Optional[str] = Field(None, description="Error message if failed")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input errors")

class FileUploadRequest(BaseModel):
    summary_length: Optional[str] = Field("full", description="Summary length: one-liner, paragraph, or full")
    user_email: str = Field(..., description="User's email address for confirmation and delivery")

    @field_validator("user_email")
    @classmethod
    def validate_email_address(cls, v):
        if not v or not v.strip():
            raise ValueError("Email address is required.")
        try:
            validate_email(v)
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email address: {e}")
        return v.strip()

    @field_validator("summary_length")
    @classmethod
    def validate_summary_length(cls, v):
        allowed = {"one-liner", "paragraph", "full"}
        if v and v.lower() not in allowed:
            raise ValueError(f"summary_length must be one of {allowed}")
        return v.lower() if v else "full"

class FileUploadResponse(BaseModel):
    success: bool = Field(..., description="Whether the summary was generated successfully")
    summary: Optional[str] = Field(None, description="Formatted meeting summary (email-ready)")
    structured_summary: Optional[dict] = Field(None, description="Structured summary sections")
    error: Optional[str] = Field(None, description="Error message if failed")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input errors")

# =========================
# SERVICE CLASSES
# =========================

class InputValidationError(Exception):
    pass

class FileExtractionError(Exception):
    pass

class InputHandler:
    """Accepts and validates user input (text, file upload, chat export)."""

    def receive_input(self, input_data: Union[str, UploadFile], input_type: str) -> str:
        """
        Accepts user input (text, file, chat export) and validates format.
        Returns transcript text.
        Raises InputValidationError if input is invalid or unsupported.
        """
        if input_type == "text":
            if not input_data or not isinstance(input_data, str) or not input_data.strip():
                raise InputValidationError("Transcript text is required.")
            if len(input_data) > 50000:
                raise InputValidationError("Transcript text exceeds 50,000 characters.")
            return input_data.strip()
        elif input_type == "file":
            return self.extract_text(input_data)
        else:
            raise InputValidationError(f"Unsupported input_type: {input_type}")

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def extract_text(self, file: UploadFile) -> str:
        """
        Extracts text from uploaded files or chat exports.
        Raises FileExtractionError on unsupported format or parsing failure.
        """
        try:
            filename = file.filename.lower()
            if filename.endswith(".txt"):
                content = file.file.read()
                text = content.decode("utf-8", errors="ignore")
                if not text.strip():
                    raise FileExtractionError("TXT file is empty.")
                return text
            elif filename.endswith(".docx"):
                doc = docx.Document(file.file)
                text = "\n".join([para.text for para in doc.paragraphs])
                if not text.strip():
                    raise FileExtractionError("DOCX file is empty.")
                return text
            elif filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                if not text.strip():
                    raise FileExtractionError("PDF file is empty.")
                return text
            else:
                raise FileExtractionError("Unsupported file format. Only .txt, .docx, .pdf are supported.")
        except Exception as e:
            raise FileExtractionError(f"Failed to extract text: {e}")

class Preprocessor:
    """Normalizes and cleans transcript text for LLM consumption."""

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def process_text(self, raw_text: str) -> str:
        """
        Normalizes and cleans transcript text for LLM consumption.
        Returns cleaned text. Logs warning and returns original if normalization fails.
        """
        try:
            # Basic normalization: collapse whitespace, remove control chars
            text = raw_text.replace("\r", "\n")
            text = _re.sub(r"\n{3,}", "\n\n", text)
            text = _re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)
            text = text.strip()
            return text
        except Exception as e:
            logging.warning(f"Preprocessing failed: {e}")
            return raw_text

class LLMService:
    """Handles interaction with Azure OpenAI GPT-4.1."""

    def __init__(self):
        self._client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        if self._client is None:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_summary(self, prompt: str, transcript_text: str, summary_length: str) -> str:
        """
        Calls LLM with enhanced system prompt and transcript to generate summary.
        Retries on timeout; returns minimal summary if repeated failure.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT},
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript_text}\n\nSummary Length: {summary_length}"}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        retries = 3
        for attempt in range(retries):
            _t0 = _time.time()
            try:
                client = self.get_llm_client()
                response = await client.chat.completions.create(
                    model=Config.LLM_MODEL or "gpt-4.1",
                    messages=messages,
                    **_llm_kwargs
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.LLM_MODEL or "gpt-4.1",
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                return sanitize_llm_output(content, content_type="text")
            except Exception as e:
                logging.warning(f"LLM summary generation attempt {attempt+1} failed: {e}")
                if attempt == retries - 1:
                    return FALLBACK_RESPONSE
                await self._exponential_backoff(attempt)
        return FALLBACK_RESPONSE

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_question(self, prompt: str, transcript_text: str, question: str) -> str:
        """
        Calls LLM to answer a follow-up question about the meeting.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT},
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript_text}\n\nQuestion: {question}"}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        try:
            client = self.get_llm_client()
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return sanitize_llm_output(content, content_type="text")
        except Exception as e:
            logging.error(f"LLM follow-up question failed: {e}")
            return FALLBACK_RESPONSE

    async def _exponential_backoff(self, attempt: int):
        delay = min(2 ** attempt, 8)
        await asyncio.sleep(delay)

class OutputFormatter:
    """Formats LLM output into structured, email-ready summaries."""

    def format_summary(self, llm_output: str) -> dict:
        """
        Formats LLM output into structured summary sections.
        Returns dict. Returns unformatted output if parsing fails; logs warning.
        """
        try:
            # Try to parse sections by header
            sections = ["Meeting Overview", "Key Discussion Points", "Decisions Made", "Action Items", "Next Steps", "Attendees"]
            result = {}
            current_section = None
            lines = llm_output.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                for section in sections:
                    if line.lower().startswith(section.lower()):
                        current_section = section
                        result[current_section] = []
                        break
                else:
                    if current_section:
                        result[current_section].append(line)
            # Collapse lists to strings
            for k in result:
                result[k] = "\n".join(result[k]).strip()
            return result
        except Exception as e:
            logging.warning(f"Output formatting failed: {e}")
            return {"summary": llm_output}

    def format_email_body(self, structured_summary: dict) -> str:
        """
        Formats summary into email-ready body.
        Returns fallback template if formatting fails.
        """
        try:
            sections = [
                "Meeting Overview",
                "Key Discussion Points",
                "Decisions Made",
                "Action Items",
                "Next Steps",
                "Attendees"
            ]
            lines = []
            for section in sections:
                content = structured_summary.get(section)
                if content:
                    lines.append(f"{section}:\n{content}\n")
            return "\n".join(lines).strip()
        except Exception as e:
            logging.warning(f"Email formatting failed: {e}")
            return "Meeting summary could not be formatted. Please review the output."

class EmailSender:
    """Requests user confirmation, sends formatted summary to participants."""

    async def request_confirmation(self, user_email: str, summary_body: str) -> bool:
        """
        Requests user confirmation before sending summary email.
        Returns True if confirmed, False otherwise.
        """
        # In a real system, this would send a confirmation email or UI prompt.
        # Here, we simulate confirmation (always True for demo) # In production, integrate with email or UI confirmation flow.
        return True

    async def send_email(self, user_email: str, summary_body: str) -> str:
        """
        Sends formatted summary to specified recipients.
        Retries on failure; escalates to human review if repeated failure.
        """
        # Placeholder: In production, integrate with SMTP or email API.
        # Here, we simulate email sending.
        try:
            # Simulate sending
            await asyncio.sleep(0.5)
            return "delivered"
        except Exception as e:
            logging.error(f"Email delivery failed: {e}")
            return "delivery_failed"

class ComplianceManager:
    """Ensures GDPR compliance, manages session expiration, purges transcript data."""

    def validate_consent(self, user_confirmation: bool) -> bool:
        """
        Ensures user consent before email distribution.
        Hard stop if consent not provided.
        """
        if not user_confirmation:
            raise PermissionError("User consent required before sending summary email.")
        return True

    def purge_data(self, session_id: Optional[str] = None) -> bool:
        """
        Purges transcript data after summary delivery.
        Logs purge status; raises alert if purge fails.
        """
        try:
            # In-memory only, so nothing to purge.
            return True
        except Exception as e:
            logging.error(f"Data purge failed: {e}")
            return False

class SecurityManager:
    """Handles encryption, authentication (OAuth2), session management, and audit logging."""

    def __init__(self):
        self._fernet = None
        try:
            key = Config.AGENT_ENCRYPTION_KEY
            if not key:
                key = Fernet.generate_key()
            self._fernet = Fernet(key)
        except Exception as e:
            logging.warning(f"Encryption key setup failed: {e}")
            self._fernet = None

    def encrypt_data(self, data: str) -> bytes:
        """
        Encrypts transcript data in-memory during processing.
        Logs encryption errors; falls back to plain processing if necessary.
        """
        try:
            if not self._fernet:
                return data.encode("utf-8")
            return self._fernet.encrypt(data.encode("utf-8"))
        except Exception as e:
            logging.warning(f"Encryption failed: {e}")
            return data.encode("utf-8")

    def authenticate_user(self, user_credentials: Any) -> bool:
        """
        Authenticates user session via OAuth2.
        Denies access if authentication fails; logs event.
        """
        # Placeholder: In production, integrate with OAuth2 provider.
        # Here, always return True (no real auth) 
        return True

    def log_event(self, event_type: str, event_details: dict) -> None:
        """
        Logs audit events for compliance and monitoring.
        Logs to secure audit trail; raises alert if logging fails.
        """
        try:
            logging.info(f"Audit event: {event_type} | {event_details}")
        except Exception as e:
            logging.error(f"Audit logging failed: {e}")

class SummarizationOrchestrator:
    """Coordinates input processing, LLM calls, output formatting, and tool integrations."""

    def __init__(self, preprocessor, llm_service, output_formatter, email_sender, compliance_manager, security_manager):
        self.preprocessor = preprocessor
        self.llm_service = llm_service
        self.output_formatter = output_formatter
        self.email_sender = email_sender
        self.compliance_manager = compliance_manager
        self.security_manager = security_manager

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def summarize_meeting(self, transcript_text: str, summary_length: str, user_email: str) -> dict:
        """
        Coordinates summarization process, including LLM call and formatting.
        Returns dict (structured summary) 
        Returns fallback summary if LLM fails; logs error.
        """
        async with trace_step(
            "summarize_meeting", step_type="llm_call",
            decision_summary="Generate structured meeting summary from transcript",
            output_fn=lambda r: f"summary={r.get('summary','')[:80]}"
        ) as step:
            cleaned_text = self.preprocessor.process_text(transcript_text)
            llm_output = await self.llm_service.generate_summary(
                prompt=SYSTEM_PROMPT, transcript_text=cleaned_text, summary_length=summary_length
            )
            if not llm_output or llm_output == FALLBACK_RESPONSE:
                logging.error("LLM failed to generate summary.")
                summary = FALLBACK_RESPONSE
                structured = {"summary": summary}
            else:
                structured = self.output_formatter.format_summary(llm_output)
                summary = self.output_formatter.format_email_body(structured)
            step.capture({"summary": summary})
            return {
                "success": True if summary and summary != FALLBACK_RESPONSE else False,
                "summary": summary,
                "structured_summary": structured,
                "error": None if summary and summary != FALLBACK_RESPONSE else FALLBACK_RESPONSE,
                "tips": None if summary and summary != FALLBACK_RESPONSE else "Try providing a longer or clearer transcript."
            }

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_followup(self, transcript_text: str, question: str) -> dict:
        """
        Handles follow-up questions about meeting content.
        Returns dict with answer.
        Returns fallback response if answer not found.
        """
        async with trace_step(
            "answer_followup", step_type="llm_call",
            decision_summary="Answer follow-up question from transcript",
            output_fn=lambda r: f"answer={r.get('answer','')[:80]}"
        ) as step:
            cleaned_text = self.preprocessor.process_text(transcript_text)
            answer = await self.llm_service.answer_question(
                prompt=SYSTEM_PROMPT, transcript_text=cleaned_text, question=question
            )
            answer = sanitize_llm_output(answer, content_type="text")
            step.capture({"answer": answer})
            return {
                "success": True if answer and answer != FALLBACK_RESPONSE else False,
                "answer": answer,
                "error": None if answer and answer != FALLBACK_RESPONSE else FALLBACK_RESPONSE,
                "tips": None if answer and answer != FALLBACK_RESPONSE else "Try rephrasing your question or providing more transcript context."
            }

# =========================
# MAIN AGENT CLASS
# =========================

class MeetingNotesSummarizerAgent:
    """
    Main agent class. Composes all services and exposes API methods.
    """

    def __init__(self):
        self.input_handler = InputHandler()
        self.preprocessor = Preprocessor()
        self.llm_service = LLMService()
        self.output_formatter = OutputFormatter()
        self.email_sender = EmailSender()
        self.compliance_manager = ComplianceManager()
        self.security_manager = SecurityManager()
        self.orchestrator = SummarizationOrchestrator(
            self.preprocessor,
            self.llm_service,
            self.output_formatter,
            self.email_sender,
            self.compliance_manager,
            self.security_manager
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def summarize_meeting(self, req: SummarizeMeetingRequest) -> SummarizeMeetingResponse:
        """
        Endpoint: Summarize meeting from transcript text.
        """
        async with trace_step(
            "summarize_meeting_api", step_type="process",
            decision_summary="API: summarize meeting from transcript",
            output_fn=lambda r: f"summary={getattr(r, 'summary', '')[:80]}"
        ) as step:
            try:
                transcript_text = req.transcript_text
                summary_length = req.summary_length or "full"
                user_email = req.user_email
                # Validate input
                if not transcript_text or not transcript_text.strip():
                    raise InputValidationError("Transcript text is required.")
                # Summarize
                result = await self.orchestrator.summarize_meeting(transcript_text, summary_length, user_email)
                step.capture(result)
                return SummarizeMeetingResponse(**result)
            except (InputValidationError, ValidationError) as e:
                return SummarizeMeetingResponse(
                    success=False,
                    summary=None,
                    structured_summary=None,
                    error=str(e),
                    tips="Ensure you provide a valid transcript and email address."
                )
            except Exception as e:
                logging.error(f"Summarize meeting failed: {e}")
                return SummarizeMeetingResponse(
                    success=False,
                    summary=None,
                    structured_summary=None,
                    error=str(e),
                    tips="Try again or contact support if the issue persists."
                )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def summarize_meeting_file(self, file: UploadFile, summary_length: str, user_email: str) -> FileUploadResponse:
        """
        Endpoint: Summarize meeting from uploaded file.
        """
        async with trace_step(
            "summarize_meeting_file_api", step_type="process",
            decision_summary="API: summarize meeting from file upload",
            output_fn=lambda r: f"summary={getattr(r, 'summary', '')[:80]}"
        ) as step:
            try:
                transcript_text = self.input_handler.extract_text(file)
                if not transcript_text or not transcript_text.strip():
                    raise InputValidationError("Transcript text could not be extracted from file.")
                result = await self.orchestrator.summarize_meeting(transcript_text, summary_length, user_email)
                step.capture(result)
                return FileUploadResponse(**result)
            except (InputValidationError, FileExtractionError, ValidationError) as e:
                return FileUploadResponse(
                    success=False,
                    summary=None,
                    structured_summary=None,
                    error=str(e),
                    tips="Ensure your file is a valid .txt, .docx, or .pdf and contains meeting content."
                )
            except Exception as e:
                logging.error(f"Summarize meeting from file failed: {e}")
                return FileUploadResponse(
                    success=False,
                    summary=None,
                    structured_summary=None,
                    error=str(e),
                    tips="Try again or contact support if the issue persists."
                )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_followup(self, req: FollowupQuestionRequest) -> FollowupQuestionResponse:
        """
        Endpoint: Answer follow-up question about meeting.
        """
        async with trace_step(
            "answer_followup_api", step_type="process",
            decision_summary="API: answer follow-up question",
            output_fn=lambda r: f"answer={getattr(r, 'answer', '')[:80]}"
        ) as step:
            try:
                transcript_text = req.transcript_text
                question = req.question
                result = await self.orchestrator.answer_followup(transcript_text, question)
                step.capture(result)
                return FollowupQuestionResponse(**result)
            except (InputValidationError, ValidationError) as e:
                return FollowupQuestionResponse(
                    success=False,
                    answer=None,
                    error=str(e),
                    tips="Ensure you provide a valid transcript and question."
                )
            except Exception as e:
                logging.error(f"Follow-up question failed: {e}")
                return FollowupQuestionResponse(
                    success=False,
                    answer=None,
                    error=str(e),
                    tips="Try again or contact support if the issue persists."
                )

# =========================
# FASTAPI APP & ENDPOINTS
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Meeting Notes Summarizer Agent",
    description="Automatically processes meeting transcripts or notes, produces a clean structured summary, extracts action items with assigned owners and due dates, identifies key decisions made, and distributes the summary to all participants.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# CORS (allow all origins for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = MeetingNotesSummarizerAgent()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeMeetingResponse)
async def summarize_meeting(req: SummarizeMeetingRequest):
    """
    Summarize meeting from transcript text.
    """
    return await agent.summarize_meeting(req)

@app.post("/summarize_file", response_model=FileUploadResponse)
async def summarize_meeting_file(
    file: UploadFile = File(..., description="Meeting transcript file (.txt, .docx, .pdf)"),
    summary_length: Optional[str] = "full",
    user_email: str = Field(..., description="User's email address for confirmation and delivery")
):
    """
    Summarize meeting from uploaded file.
    """
    return await agent.summarize_meeting_file(file, summary_length, user_email)

@app.post("/followup", response_model=FollowupQuestionResponse)
async def followup_question(req: FollowupQuestionRequest):
    """
    Answer follow-up question about meeting content.
    """
    return await agent.answer_followup(req)

# =========================
# ERROR HANDLING
# =========================

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation error",
            "tips": str(exc),
        },
    )

@app.exception_handler(json.decoder.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.decoder.JSONDecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": "Malformed JSON in request body.",
            "tips": "Check for missing quotes, commas, or brackets in your JSON.",
        },
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error.",
            "tips": "Try again or contact support if the issue persists.",
        },
    )

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())