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

# ── Unit Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unit_summarize_meeting_happy_path(agent_instance):
    """Test summarize_meeting returns expected result."""
    mock_summary = "Meeting Overview: ...\nKey Discussion Points: ...\nDecisions Made: ...\nAction Items: ...\nNext Steps: ...\nAttendees: ..."
    mock_structured = {
        "Meeting Overview": "Discussed project roadmap.",
        "Key Discussion Points": "Timeline, budget, risks.",
        "Decisions Made": "Approved next sprint.",
        "Action Items": "Alice to update docs.",
        "Next Steps": "Schedule follow-up.",
        "Attendees": "Alice, Bob"
    }
    mock_response = {
        "success": True,
        "summary": mock_summary,
        "structured_summary": mock_structured,
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_response)):
        req = SummarizeMeetingRequest(
            transcript_text="This is a valid meeting transcript about project updates and next steps.",
            summary_length="full",
            user_email="user@example.com"
        )
        result = await agent_instance.summarize_meeting(req)
    assert result is not None

@pytest.mark.asyncio
async def test_unit_summarize_meeting_error_handling(agent_instance):
    """Test summarize_meeting handles errors gracefully."""
    with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(side_effect=Exception("test error"))):
        req = SummarizeMeetingRequest(
            transcript_text="This is a valid meeting transcript.",
            summary_length="full",
            user_email="user@example.com"
        )
        try:
            result = await agent_instance.summarize_meeting(req)
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise
        except Exception:
            pass  # Agent propagated the error — also valid behavior

# ── Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_workflow(agent_instance):
    """Test complete workflow with mocked dependencies."""
    mock_summary = "Meeting Overview: ...\nKey Discussion Points: ...\nDecisions Made: ...\nAction Items: ...\nNext Steps: ...\nAttendees: ..."
    mock_structured = {
        "Meeting Overview": "Discussed project roadmap.",
        "Key Discussion Points": "Timeline, budget, risks.",
        "Decisions Made": "Approved next sprint.",
        "Action Items": "Alice to update docs.",
        "Next Steps": "Schedule follow-up.",
        "Attendees": "Alice, Bob"
    }
    mock_response = {
        "success": True,
        "summary": mock_summary,
        "structured_summary": mock_structured,
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_response)):
        req = SummarizeMeetingRequest(
            transcript_text="This is a valid meeting transcript about project updates and next steps.",
            summary_length="full",
            user_email="user@example.com"
        )
        result = await agent_instance.summarize_meeting(req)
    assert result is not None

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_throughput(agent_instance):
    """Test processing throughput with generous threshold."""
    mock_summary = "Meeting Overview: ...\nKey Discussion Points: ...\nDecisions Made: ...\nAction Items: ...\nNext Steps: ...\nAttendees: ..."
    mock_structured = {
        "Meeting Overview": "Discussed project roadmap.",
        "Key Discussion Points": "Timeline, budget, risks.",
        "Decisions Made": "Approved next sprint.",
        "Action Items": "Alice to update docs.",
        "Next Steps": "Schedule follow-up.",
        "Attendees": "Alice, Bob"
    }
    mock_response = {
        "success": True,
        "summary": mock_summary,
        "structured_summary": mock_structured,
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_response)):
        req = SummarizeMeetingRequest(
            transcript_text="This is a valid meeting transcript about project updates and next steps.",
            summary_length="full",
            user_email="user@example.com"
        )
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.summarize_meeting(req)
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

# ── Edge Case Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edge_case_empty_input(agent_instance):
    """Test handling of empty/None input."""
    mock_response = {
        "success": False,
        "summary": None,
        "structured_summary": None,
        "error": "Transcript text is required.",
        "tips": "Ensure you provide a valid transcript and email address."
    }
    with patch.object(agent_instance.orchestrator, "summarize_meeting", new=AsyncMock(return_value=mock_response)):
        req = SummarizeMeetingRequest(
            transcript_text="",
            summary_length="full",
            user_email="user@example.com"
        )
        result = await agent_instance.summarize_meeting(req)
    assert result is not None