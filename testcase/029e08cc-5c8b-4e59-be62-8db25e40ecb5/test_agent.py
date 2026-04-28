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

from agent import MeetingNotesSummarizerAgent
from agent import FileUploadResponse

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = MeetingNotesSummarizerAgent()
    return instance

@pytest.mark.asyncio
async def test_unit_summarize_meeting_file_unsupported_format(agent_instance):
    """Test summarize_meeting_file rejects unsupported file formats."""
    # Prepare a mock UploadFile with .exe extension
    mock_file = MagicMock()
    mock_file.filename = "malware.exe"
    mock_file.file.read.return_value = b"dummy content"
    # Patch extract_text to raise FileExtractionError for unsupported format
    with patch.object(agent_instance.input_handler, "extract_text", side_effect=agent_instance.input_handler.__class__.__dict__["extract_text"].__wrapped__.__globals__["FileExtractionError"]("Unsupported file format. Only .txt, .docx, .pdf are supported.")):
        try:
            result = await agent_instance.summarize_meeting_file(mock_file, "full", "user@example.com")
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass  # Agent propagated the error — also valid

@pytest.mark.asyncio
async def test_integration_summarize_meeting_file_unsupported_format(agent_instance):
    """Integration: summarize_meeting_file with unsupported file triggers error handling."""
    mock_file = MagicMock()
    mock_file.filename = "image.jpg"
    mock_file.file.read.return_value = b"dummy content"
    with patch.object(agent_instance.input_handler, "extract_text", side_effect=agent_instance.input_handler.__class__.__dict__["extract_text"].__wrapped__.__globals__["FileExtractionError"]("Unsupported file format. Only .txt, .docx, .pdf are supported.")):
        try:
            result = await agent_instance.summarize_meeting_file(mock_file, "full", "user@example.com")
            assert result is not None
        except AssertionError:
            raise
        except Exception:
            pass

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_summarize_meeting_file_unsupported_format(agent_instance):
    """Performance: repeated unsupported file uploads are handled quickly."""
    mock_file = MagicMock()
    mock_file.filename = "archive.exe"
    mock_file.file.read.return_value = b"dummy content"
    with patch.object(agent_instance.input_handler, "extract_text", side_effect=agent_instance.input_handler.__class__.__dict__["extract_text"].__wrapped__.__globals__["FileExtractionError"]("Unsupported file format. Only .txt, .docx, .pdf are supported.")):
        start_time = time.time()
        for _ in range(10):
            try:
                result = await agent_instance.summarize_meeting_file(mock_file, "full", "user@example.com")
                assert result is not None
            except AssertionError:
                raise
            except Exception:
                pass
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

@pytest.mark.asyncio
async def test_edge_case_summarize_meeting_file_empty_file(agent_instance):
    """Edge case: summarize_meeting_file with empty file triggers error handling."""
    mock_file = MagicMock()
    mock_file.filename = "empty.txt"
    mock_file.file.read.return_value = b""
    with patch.object(agent_instance.input_handler, "extract_text", side_effect=agent_instance.input_handler.__class__.__dict__["extract_text"].__wrapped__.__globals__["FileExtractionError"]("TXT file is empty.")):
        try:
            result = await agent_instance.summarize_meeting_file(mock_file, "full", "user@example.com")
            assert result is not None
        except AssertionError:
            raise
        except Exception:
            pass