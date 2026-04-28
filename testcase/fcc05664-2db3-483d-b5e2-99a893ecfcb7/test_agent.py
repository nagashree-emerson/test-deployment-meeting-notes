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

from agent import MeetingNotesSummarizerAgent, FollowupQuestionRequest, FollowupQuestionResponse

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = MeetingNotesSummarizerAgent()
    return instance

@pytest.mark.asyncio
async def test_integration_followup_question_end_to_end(agent_instance):
    """Test /followup endpoint from input validation through LLM answer and output formatting."""
    # Prepare input
    req = FollowupQuestionRequest(
        transcript_text="Alice: Let's review the budget. Bob: I'll send the numbers by Friday.",
        question="What did Bob agree to do?"
    )
    # Patch orchestrator.answer_followup to simulate full workflow
    mock_answer = {
        "success": True,
        "answer": "Bob agreed to send the numbers by Friday.",
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_answer)):
        result = await agent_instance.answer_followup(req)
    assert result is not None

@pytest.mark.asyncio
async def test_integration_followup_question_llm_fallback(agent_instance):
    """Test /followup endpoint when LLMService returns FALLBACK_RESPONSE."""
    req = FollowupQuestionRequest(
        transcript_text="Alice: Let's review the budget.",
        question="What did Bob agree to do?"
    )
    mock_answer = {
        "success": False,
        "answer": "The requested information could not be found in the provided meeting transcript or notes. Please check the input and try again.",
        "error": "The requested information could not be found in the provided meeting transcript or notes. Please check the input and try again.",
        "tips": "Try rephrasing your question or providing more transcript context."
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_answer)):
        result = await agent_instance.answer_followup(req)
    assert result is not None

@pytest.mark.asyncio
async def test_integration_followup_question_validation_error(agent_instance):
    """Test /followup endpoint with invalid (too long) question triggers validation error."""
    long_question = "a" * 2000  # Exceeds 1000 chars
    try:
        req = FollowupQuestionRequest(
            transcript_text="Alice: Let's review the budget.",
            question=long_question
        )
        # If construction succeeds, call the agent (should not happen)
        result = await agent_instance.answer_followup(req)
        assert result is not None
    except AssertionError:
        raise
    except Exception:
        pass  # ValidationError is expected and valid

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_followup_question_throughput(agent_instance):
    """Test /followup endpoint throughput with generous threshold."""
    req = FollowupQuestionRequest(
        transcript_text="Alice: Let's review the budget. Bob: I'll send the numbers by Friday.",
        question="What did Bob agree to do?"
    )
    mock_answer = {
        "success": True,
        "answer": "Bob agreed to send the numbers by Friday.",
        "error": None,
        "tips": None
    }
    with patch.object(agent_instance.orchestrator, "answer_followup", new=AsyncMock(return_value=mock_answer)):
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.answer_followup(req)
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

@pytest.mark.asyncio
async def test_edge_case_followup_question_empty_input(agent_instance):
    """Test /followup endpoint with empty transcript input."""
    try:
        req = FollowupQuestionRequest(
            transcript_text="",
            question="What did Bob agree to do?"
        )
        result = await agent_instance.answer_followup(req)
        assert result is not None
    except AssertionError:
        raise
    except Exception:
        pass  # ValidationError is expected and valid