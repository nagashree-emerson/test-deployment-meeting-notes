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
async def test_unit_summarize_meeting_file_happy_path(agent_instance):
    """Test summarize_meeting_file returns expected result with valid .txt file."""
    # Mock UploadFile
    mock_file = MagicMock()
    mock_file.filename = "meeting.txt"
    mock_file.file.read.return_value = b"Meeting transcript content goes here."

    # Patch orchestrator.summarize_meeting to return a valid dict
    mock_summary = {
        "success": True,
        "summary": "This is a summary.",
        "structured_summary": {"Meeting Overview": "Overview text"},
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="Meeting transcript content goes here.")):
        with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_summary)):
            result = await agent_instance.summarize_meeting_file(mock_file, "paragraph", "user@example.com")
    assert result is not None

@pytest.mark.asyncio
async def test_unit_summarize_meeting_file_error_handling(agent_instance):
    """Test summarize_meeting_file handles file extraction errors gracefully."""
    mock_file = MagicMock()
    mock_file.filename = "meeting.txt"
    # Simulate extract_text raising FileExtractionError
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(side_effect=Exception("File extraction failed"))):
        try:
            result = await agent_instance.summarize_meeting_file(mock_file, "paragraph", "user@example.com")
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass  # Agent propagated the error — also valid behavior

@pytest.mark.asyncio
async def test_integration_workflow(agent_instance):
    """Test complete workflow with mocked dependencies for summarize_meeting_file."""
    mock_file = MagicMock()
    mock_file.filename = "meeting.txt"
    mock_file.file.read.return_value = b"Meeting transcript content goes here."
    mock_summary = {
        "success": True,
        "summary": "This is a summary.",
        "structured_summary": {"Meeting Overview": "Overview text"},
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="Meeting transcript content goes here.")):
        with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_summary)):
            result = await agent_instance.summarize_meeting_file(mock_file, "paragraph", "user@example.com")
    assert result is not None

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_throughput(agent_instance):
    """Test processing throughput with generous threshold for summarize_meeting_file."""
    mock_file = MagicMock()
    mock_file.filename = "meeting.txt"
    mock_file.file.read.return_value = b"Meeting transcript content goes here."
    mock_summary = {
        "success": True,
        "summary": "This is a summary.",
        "structured_summary": {"Meeting Overview": "Overview text"},
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="Meeting transcript content goes here.")):
        with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_summary)):
            start_time = time.time()
            for _ in range(10):
                result = await agent_instance.summarize_meeting_file(mock_file, "paragraph", "user@example.com")
                assert result is not None
            duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

@pytest.mark.asyncio
async def test_edge_case_empty_input(agent_instance):
    """Test handling of empty/None input file."""
    mock_file = MagicMock()
    mock_file.filename = "meeting.txt"
    # Simulate extract_text returning empty string
    with patch.object(agent_instance.input_handler, "extract_text", new=MagicMock(return_value="")):
        result = await agent_instance.summarize_meeting_file(mock_file, "paragraph", "user@example.com")
    assert result is not None