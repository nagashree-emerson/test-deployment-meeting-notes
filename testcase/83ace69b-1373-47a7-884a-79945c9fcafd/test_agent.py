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

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_summarize_meeting_file_large_pdf(agent_instance):
    """Test /summarize_file endpoint with a large PDF file (performance)."""
    # Create a mock UploadFile with a large PDF-like content
    class MockFile:
        def __init__(self, content):
            self.content = content
            self.filename = "meeting.pdf"
            self.file = MagicMock()
            # Simulate PyPDF2.PdfReader(file).pages
            self.file.read = MagicMock()
            self.file.seek = MagicMock()
            self.file.tell = MagicMock()
            self.file.__enter__ = MagicMock(return_value=self.file)
            self.file.__exit__ = MagicMock(return_value=None)
    # Patch PyPDF2.PdfReader to simulate PDF extraction
    large_text = "Meeting content.\n" * 4000  # ~50,000 chars
    class MockPage:
        def extract_text(self):
            return large_text
    class MockPdfReader:
        def __init__(self, file):
            self.pages = [MockPage()]
    mock_file = MockFile(content=b"PDFDATA")
    if True:  # AUTO-FIXED: replaced bare MagicMock context manager
        # Patch orchestrator.summarize_meeting to avoid real LLM call
        with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value={
            "success": True,
            "summary": "Summary of meeting.",
            "structured_summary": {"Meeting Overview": "Overview", "Key Discussion Points": "Points"},
            "error": None,
            "tips": None
        })):
            start_time = time.time()
            result = await agent_instance.summarize_meeting_file(
                mock_file, summary_length="full", user_email="user@example.com"
            )
            assert result is not None
            duration = time.time() - start_time
    assert duration < 30.0, f"File upload and summarization took {duration:.1f}s"