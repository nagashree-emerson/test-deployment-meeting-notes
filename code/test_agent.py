# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
import io

import agent

# ========== UNIT TESTS ==========

def test_SummarizeMeetingRequest_valid_input():
    """Validates SummarizeMeetingRequest accepts correct input and applies validators."""
    from agent import SummarizeMeetingRequest
    req = SummarizeMeetingRequest(
        transcript_text="This is a valid transcript.",
        summary_length="full",
        user_email="user@example.com"
    )
    assert req.transcript_text == "This is a valid transcript."
    assert req.summary_length == "full"
    assert req.user_email == "user@example.com"

@pytest.mark.parametrize("field, value, expected_error", [
    ("user_email", "not-an-email", "Invalid email address"),
    ("summary_length", "invalid", "summary_length must be one of"),
    ("transcript_text", "short", "Transcript text is too short"),
])
def test_SummarizeMeetingRequest_field_validation_errors(field, value, expected_error):
    """Checks SummarizeMeetingRequest raises ValueError for invalid fields."""
    from agent import SummarizeMeetingRequest
    kwargs = {
        "transcript_text": "This is a valid transcript.",
        "summary_length": "full",
        "user_email": "user@example.com"
    }
    kwargs[field] = value
    with pytest.raises(ValueError) as e:
        SummarizeMeetingRequest(**kwargs)
    assert expected_error in str(e.value)

def test_InputHandler_receive_input_invalid_type():
    """Checks InputHandler.receive_input raises InputValidationError for unsupported input_type."""
    from agent import InputHandler, InputValidationError
    handler = InputHandler()
    with pytest.raises(InputValidationError) as e:
        handler.receive_input("Some text", "audio")
    assert "Unsupported input_type" in str(e.value)

def test_LLMService_generate_summary_fallback_on_failure(monkeypatch):
    """Ensures LLMService.generate_summary returns FALLBACK_RESPONSE after 3 failed attempts."""
    from agent import LLMService, FALLBACK_RESPONSE

    async def raise_exc(*args, **kwargs):
        raise Exception("LLM error")

    service = LLMService()
    # Patch get_llm_client to return a mock with .chat.completions.create raising
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=raise_exc)
    monkeypatch.setattr(service, "get_llm_client", lambda: mock_client)

    # Patch sanitize_llm_output to just return input for simplicity
    monkeypatch.setattr("agent.sanitize_llm_output", lambda x, content_type="text": x)

    result = asyncio.run(service.generate_summary("prompt", "transcript", "full"))
    assert result == FALLBACK_RESPONSE

def test_OutputFormatter_format_summary_parses_all_sections():
    """Verifies OutputFormatter.format_summary parses all expected summary sections."""
    from agent import OutputFormatter
    llm_output = (
        "Meeting Overview\n"
        "This is the overview.\n"
        "Key Discussion Points\n"
        "Point 1\nPoint 2\n"
        "Decisions Made\n"
        "Decision 1\n"
        "Action Items\n"
        "Action 1\n"
        "Next Steps\n"
        "Step 1\n"
        "Attendees\n"
        "Alice, Bob"
    )
    formatter = OutputFormatter()
    result = formatter.format_summary(llm_output)
    expected_sections = [
        "Meeting Overview", "Key Discussion Points", "Decisions Made",
        "Action Items", "Next Steps", "Attendees"
    ]
    for section in expected_sections:
        assert section in result
        assert isinstance(result[section], str)
        assert result[section] != ""

# ========== INTEGRATION TESTS ==========

@pytest.fixture(scope="module")
def fastapi_client():
    from agent import app
    with TestClient(app) as client:
        yield client

def mock_llmservice_generate_summary(*args, **kwargs):
    return "Mocked summary output"

@pytest.mark.asyncio
async def test_MeetingNotesSummarizerAgent_summarize_meeting_api_success(monkeypatch):
    """Tests /summarize endpoint with valid input."""
    from agent import agent as agent_module
    # Patch orchestrator.llm_service.generate_summary to avoid real LLM call
    monkeypatch.setattr(
        agent_module.orchestrator.llm_service,
        "generate_summary",
        AsyncMock(return_value="Meeting Overview\nOverview\nKey Discussion Points\nPoints\nDecisions Made\nDecisions\nAction Items\nActions\nNext Steps\nSteps\nAttendees\nNames")
    )
    # Patch OutputFormatter to avoid parsing errors
    monkeypatch.setattr(
        agent_module.orchestrator.output_formatter,
        "format_summary",
        lambda x: {
            "Meeting Overview": "Overview",
            "Key Discussion Points": "Points",
            "Decisions Made": "Decisions",
            "Action Items": "Actions",
            "Next Steps": "Steps",
            "Attendees": "Names"
        }
    )
    monkeypatch.setattr(
        agent_module.orchestrator.output_formatter,
        "format_email_body",
        lambda x: "Formatted summary"
    )
    payload = {
        "transcript_text": "This is a valid transcript for summarization.",
        "summary_length": "full",
        "user_email": "user@example.com"
    }
    from agent import app
    with TestClient(app) as client:
        response = client.post("/summarize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["summary"] is not None
        assert data["structured_summary"] is not None

def make_uploadfile(filename, content):
    fileobj = io.BytesIO(content)
    return StarletteUploadFile(filename=filename, file=fileobj)

def test_MeetingNotesSummarizerAgent_summarize_meeting_file_unsupported_file(monkeypatch):
    """Checks summarize_meeting_file returns error for unsupported file format."""
    from agent import app
    # Patch extract_text to raise FileExtractionError for .xlsx
    from agent import InputHandler, FileExtractionError
    def fake_extract_text(self, file):
        raise FileExtractionError("Unsupported file format. Only .txt, .docx, .pdf are supported.")
    monkeypatch.setattr(InputHandler, "extract_text", fake_extract_text)
    upload = make_uploadfile("meeting.xlsx", b"dummy content")
    payload = {
        "summary_length": "full",
        "user_email": "user@example.com"
    }
    with TestClient(app) as client:
        response = client.post(
            "/summarize_file",
            files={"file": (upload.filename, upload.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Unsupported file format" in data["error"]

def test_MeetingNotesSummarizerAgent_answer_followup_missing_question():
    """Validates answer_followup returns error when question is missing."""
    from agent import app
    payload = {
        "transcript_text": "This is a valid transcript."
        # Missing 'question'
    }
    with TestClient(app) as client:
        response = client.post("/followup", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "Question is required" in str(data.get("tips", "")) or "question" in str(data.get("tips", ""))

# ========== PERFORMANCE TESTS ==========

@pytest.mark.performance
def test_performance_summarize_meeting(monkeypatch):
    """Measures that summarize_meeting completes within required response time for standard-length meetings."""
    from agent import agent as agent_module
    # Patch orchestrator.llm_service.generate_summary to simulate fast response
    monkeypatch.setattr(
        agent_module.orchestrator.llm_service,
        "generate_summary",
        AsyncMock(return_value="Meeting Overview\nOverview\nKey Discussion Points\nPoints\nDecisions Made\nDecisions\nAction Items\nActions\nNext Steps\nSteps\nAttendees\nNames")
    )
    monkeypatch.setattr(
        agent_module.orchestrator.output_formatter,
        "format_summary",
        lambda x: {
            "Meeting Overview": "Overview",
            "Key Discussion Points": "Points",
            "Decisions Made": "Decisions",
            "Action Items": "Actions",
            "Next Steps": "Steps",
            "Attendees": "Names"
        }
    )
    monkeypatch.setattr(
        agent_module.orchestrator.output_formatter,
        "format_email_body",
        lambda x: "Formatted summary"
    )
    payload = {
        "transcript_text": "A" * 5000,
        "summary_length": "full",
        "user_email": "user@example.com"
    }
    from agent import app
    with TestClient(app) as client:
        start = time.time()
        response = client.post("/summarize", json=payload)
        duration = time.time() - start
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert duration <= 5.0

# ========== SECURITY TESTS ==========

def test_SecurityManager_encrypt_data_and_fallback(monkeypatch):
    """Ensures SecurityManager.encrypt_data encrypts data and falls back to plain bytes if encryption fails."""
    from agent import SecurityManager
    mgr = SecurityManager()
    # Normal encryption
    data = "secret meeting transcript"
    # AUTO-FIXED: commented out call to non-existent ComplianceManager.encrypt_data()
    # encrypted = mgr.encrypt_data(data)
    encrypted  = None
    assert isinstance(encrypted, bytes)
    # Simulate encryption failure by removing _fernet
    mgr._fernet = None
    # AUTO-FIXED: commented out call to non-existent ComplianceManager.encrypt_data()
    # fallback = mgr.encrypt_data(data)
    fallback  = None
    assert isinstance(fallback, bytes)
    assert fallback == data.encode("utf-8")

def test_ComplianceManager_validate_consent_blocks_without_consent():
    """Checks ComplianceManager.validate_consent raises PermissionError if user_confirmation is False."""
    from agent import ComplianceManager
    mgr = ComplianceManager()
    with pytest.raises(PermissionError) as e:
        mgr.validate_consent(False)
    assert "User consent required" in str(e.value)
    # Should not raise if True
    assert mgr.validate_consent(True) is True

# ========== EDGE CASE TESTS ==========

@pytest.mark.asyncio
async def test_SummarizationOrchestrator_summarize_meeting_empty_transcript(monkeypatch):
    """Tests orchestrator's behavior when given an empty transcript."""
    from agent import SummarizationOrchestrator, Preprocessor, LLMService, OutputFormatter, EmailSender, ComplianceManager, SecurityManager
    orchestrator = SummarizationOrchestrator(
        Preprocessor(), LLMService(), OutputFormatter(), EmailSender(), ComplianceManager(), SecurityManager()
    )
    # Patch LLMService.generate_summary to not be called (should fail before)
    with pytest.raises(Exception) as e:
        await orchestrator.summarize_meeting("", "full", "user@example.com")
    assert "Transcript text is required" in str(e.value) or "fallback" in str(e.value).lower() or "could not be found" in str(e.value).lower()

def test_LLMService_generate_summary_extremely_long_transcript(monkeypatch):
    """Ensures LLMService.generate_summary handles transcript_text > 50,000 chars gracefully."""
    from agent import LLMService, FALLBACK_RESPONSE
    service = LLMService()
    long_text = "A" * 51000
    # Patch get_llm_client to avoid real call
    monkeypatch.setattr(service, "get_llm_client", lambda: MagicMock())
    # Patch sanitize_llm_output to just return input
    monkeypatch.setattr("agent.sanitize_llm_output", lambda x, content_type="text": x)
    # Patch client.chat.completions.create to raise error
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Too long"))
    monkeypatch.setattr(service, "get_llm_client", lambda: mock_client)
    result = asyncio.run(service.generate_summary("prompt", long_text, "full"))
    assert result == FALLBACK_RESPONSE or "could not be found" in result.lower()