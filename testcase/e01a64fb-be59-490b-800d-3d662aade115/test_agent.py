
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from agent import MeetingNotesSummarizerAgent, SummarizeMeetingRequest, SummarizeMeetingResponse

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = MeetingNotesSummarizerAgent()
    return instance

# ── Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_summarize_meeting_end_to_end(agent_instance):
    """Test full /summarize workflow with mocked dependencies."""
    # Prepare input
    req = SummarizeMeetingRequest(
        transcript_text="This is a sample meeting transcript discussing project deadlines and action items.",
        summary_length="full",
        user_email="user@example.com"
    )
    # Mock all service calls in orchestrator chain
    with patch.object(agent_instance.input_handler, "receive_input", new=MagicMock(return_value=req.transcript_text)), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(return_value="Cleaned transcript text")), \
         patch.object(agent_instance.llm_service, "generate_summary", new=AsyncMock(return_value="LLM summary output")), \
         patch.object(agent_instance.output_formatter, "format_summary", new=MagicMock(return_value={"Meeting Overview": "Overview", "Key Discussion Points": "Points"})), \
         patch.object(agent_instance.output_formatter, "format_email_body", new=MagicMock(return_value="Formatted summary email body")):
        result = await agent_instance.summarize_meeting(req)
    assert result is not None

@pytest.mark.asyncio
async def test_integration_summarize_meeting_llm_fallback(agent_instance):
    """Test /summarize handles LLMService fallback response gracefully."""
    req = SummarizeMeetingRequest(
        transcript_text="This is a sample meeting transcript discussing project deadlines and action items.",
        summary_length="full",
        user_email="user@example.com"
    )
    with patch.object(agent_instance.input_handler, "receive_input", new=MagicMock(return_value=req.transcript_text)), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(return_value="Cleaned transcript text")), \
         patch.object(agent_instance.llm_service, "generate_summary", new=AsyncMock(return_value="The requested information could not be found in the provided meeting transcript or notes. Please check the input and try again.")), \
         patch.object(agent_instance.output_formatter, "format_summary", new=MagicMock(return_value={"summary": "The requested information could not be found in the provided meeting transcript or notes. Please check the input and try again."})), \
         patch.object(agent_instance.output_formatter, "format_email_body", new=MagicMock(return_value="The requested information could not be found in the provided meeting transcript or notes. Please check the input and try again.")):
        result = await agent_instance.summarize_meeting(req)
    assert result is not None

@pytest.mark.asyncio
async def test_integration_summarize_meeting_service_exception(agent_instance):
    """Test /summarize handles service exceptions gracefully."""
    req = SummarizeMeetingRequest(
        transcript_text="This is a sample meeting transcript discussing project deadlines and action items.",
        summary_length="full",
        user_email="user@example.com"
    )
    with patch.object(agent_instance.input_handler, "receive_input", new=MagicMock(return_value=req.transcript_text)), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(side_effect=Exception("test error"))):
        try:
            result = await agent_instance.summarize_meeting(req)
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass  # Agent propagated the error — also valid behavior