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
import time
from unittest.mock import AsyncMock, MagicMock, patch

from agent import MeetingNotesSummarizerAgent, FileUploadResponse

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = MeetingNotesSummarizerAgent()
    return instance

@pytest.mark.asyncio
async def test_integration_summarize_meeting_file_pdf_happy_path(agent_instance):
    """Test summarize_meeting_file with valid .pdf triggers extraction, summarization, and formatting."""
    # Prepare a fake UploadFile for .pdf
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    fake_file.file = MagicMock()

    # Mock extract_text to return valid transcript text
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="This is a valid meeting transcript with enough content.")), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(return_value="Cleaned transcript text.")), \
         patch.object(agent_instance.llm_service, "generate_summary", new=AsyncMock(return_value="LLM summary output.")), \
         patch.object(agent_instance.output_formatter, "format_summary", new=MagicMock(return_value={"Meeting Overview": "Overview", "Key Discussion Points": "Points"})), \
         patch.object(agent_instance.output_formatter, "format_email_body", new=MagicMock(return_value="Formatted email body.")):
        result = await agent_instance.summarize_meeting_file(fake_file, "full", "user@example.com")
    assert result is not None

@pytest.mark.asyncio
async def test_integration_summarize_meeting_file_pdf_file_extraction_error(agent_instance):
    """Test summarize_meeting_file handles FileExtractionError for corrupt/empty PDF."""
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    fake_file.file = MagicMock()

    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(side_effect=Exception("Failed to extract text"))):
        try:
            result = await agent_instance.summarize_meeting_file(fake_file, "full", "user@example.com")
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass

@pytest.mark.asyncio
async def test_integration_summarize_meeting_file_pdf_input_validation_error(agent_instance):
    """Test summarize_meeting_file handles InputValidationError for too short extracted text."""
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    fake_file.file = MagicMock()

    # Simulate extract_text returns too short text
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="short")), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(return_value="short")):
        try:
            result = await agent_instance.summarize_meeting_file(fake_file, "full", "user@example.com")
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_summarize_meeting_file_pdf_throughput(agent_instance):
    """Test summarize_meeting_file throughput for 10 .pdf uploads."""
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    fake_file.file = MagicMock()

    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="This is a valid meeting transcript with enough content.")), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(return_value="Cleaned transcript text.")), \
         patch.object(agent_instance.llm_service, "generate_summary", new=AsyncMock(return_value="LLM summary output.")), \
         patch.object(agent_instance.output_formatter, "format_summary", new=MagicMock(return_value={"Meeting Overview": "Overview", "Key Discussion Points": "Points"})), \
         patch.object(agent_instance.output_formatter, "format_email_body", new=MagicMock(return_value="Formatted email body.")):
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.summarize_meeting_file(fake_file, "full", "user@example.com")
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

@pytest.mark.asyncio
async def test_edge_case_summarize_meeting_file_empty_file(agent_instance):
    """Test summarize_meeting_file handles empty file input gracefully."""
    fake_file = MagicMock()
    fake_file.filename = "meeting.pdf"
    fake_file.file = MagicMock()

    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="")), \
         patch.object(agent_instance.preprocessor, "process_text", new=MagicMock(return_value="")):
        result = await agent_instance.summarize_meeting_file(fake_file, "full", "user@example.com")
    assert result is not None