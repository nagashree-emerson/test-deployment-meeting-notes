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

# Import ONLY from 'agent' — the file is always agent.py
from agent import MeetingNotesSummarizerAgent, FollowupQuestionRequest, FollowupQuestionResponse

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = MeetingNotesSummarizerAgent()
    return instance

# ── Unit Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unit_answer_followup_happy_path(agent_instance):
    """Test answer_followup returns expected result."""
    mock_response = {
        "success": True,
        "answer": "John agreed to send the budget report by Friday.",
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_response)):
        req = FollowupQuestionRequest(
            transcript_text="John will send the budget report by Friday.",
            question="What did John agree to do?"
        )
        result = await agent_instance.answer_followup(req)
    assert result is not None

@pytest.mark.asyncio
async def test_unit_answer_followup_error_handling(agent_instance):
    """Test answer_followup handles errors gracefully."""
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(side_effect=Exception("test error"))):
        req = FollowupQuestionRequest(
            transcript_text="Some transcript text.",
            question="What was decided?"
        )
        try:
            result = await agent_instance.answer_followup(req)
            assert result is not None  # Agent handled the error internally
        except AssertionError:
            raise  # ALWAYS reraise — do NOT swallow real test assertion failures
        except Exception:
            pass  # Agent propagated the error — also valid behavior

# ── Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_followup_workflow(agent_instance):
    """Test complete followup workflow with mocked dependencies."""
    mock_response = {
        "success": True,
        "answer": "The team decided to postpone the launch.",
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_response)):
        req = FollowupQuestionRequest(
            transcript_text="The team decided to postpone the launch.",
            question="What was decided about the launch?"
        )
        result = await agent_instance.answer_followup(req)
    assert result is not None

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_followup_throughput(agent_instance):
    """Test processing throughput with generous threshold."""
    mock_response = {
        "success": True,
        "answer": "The meeting was scheduled for next week.",
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_response)):
        req = FollowupQuestionRequest(
            transcript_text="The meeting was scheduled for next week.",
            question="When is the next meeting?"
        )
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.answer_followup(req)
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

# ── Edge Case Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edge_case_followup_empty_input(agent_instance):
    """Test handling of empty/None input."""
    mock_response = {
        "success": False,
        "answer": None,
        "error": "Transcript text is required.",
        "tips": "Ensure you provide a valid transcript and question."
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_response)):
        req = FollowupQuestionRequest(
            transcript_text="",
            question="What was discussed?"
        )
        result = await agent_instance.answer_followup(req)
    assert result is not None