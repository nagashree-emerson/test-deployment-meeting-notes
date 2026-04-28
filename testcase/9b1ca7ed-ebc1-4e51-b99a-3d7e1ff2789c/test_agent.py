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

# Import ONLY from 'agent'
from agent import MeetingNotesSummarizerAgent, SummarizeMeetingRequest, SummarizeMeetingResponse

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = MeetingNotesSummarizerAgent()
    return instance

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_summarize_meeting_response_time(agent_instance):
    """Test processing throughput with generous threshold."""
    # Prepare a medium-length transcript (~1000 words)
    transcript = "This is a sample meeting transcript. " * 50  # ~500 words
    transcript += "Discussion continued on the project timeline and deliverables. " * 20  # +200 words
    transcript += "Action items were assigned to various team members. " * 10  # +100 words
    transcript += "The meeting concluded with next steps and deadlines. " * 10  # +100 words
    transcript = transcript.strip()
    # Ensure it's at least ~1000 words
    transcript = transcript + " Additional notes." * ((1000 - len(transcript.split())) // 2)

    req = SummarizeMeetingRequest(
        transcript_text=transcript,
        summary_length="full",
        user_email="user@example.com"
    )

    # Patch the orchestrator's summarize_meeting to return quickly
    mock_response = {
        "success": True,
        "summary": "Meeting summary...",
        "structured_summary": {"Meeting Overview": "Overview...", "Key Discussion Points": "Points..."},
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_response)):
        start_time = time.time()
        result = await agent_instance.summarize_meeting(req)
        assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"SummarizeMeetingResponse took {duration:.2f}s"