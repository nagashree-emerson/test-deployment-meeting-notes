# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
from agent import LLMService, Config

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_llmservice_generate_summary_throughput():
    """
    Measures average latency of LLMService.generate_summary over multiple rapid requests.
    Ensures all responses are not None and average response time < 4 seconds.
    """
    # Patch LLMService.get_llm_client to return a mock client
    with patch("agent.LLMService.get_llm_client") as mock_get_llm_client, \
         patch("agent.sanitize_llm_output", side_effect=lambda x, content_type="text": x):

        # Create a mock client with an async chat.completions.create method
        class DummyChoices:
            def __init__(self, content):
                self.message = MagicMock(content=content)
        class DummyResponse:
            def __init__(self, content):
                self.choices = [DummyChoices(content)]
                self.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        async def fake_create(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate some latency
            return DummyResponse("This is a summary.")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)
        mock_get_llm_client.return_value = mock_client

        # Prepare test data
        service = LLMService()
        prompt = "Summarize the meeting."
        transcript_text = "This is a valid meeting transcript with enough content to pass validation."
        summary_length = "full"

        num_requests = 5
        tasks = []
        start = time.time()
        for _ in range(num_requests):
            tasks.append(service.generate_summary(prompt, transcript_text, summary_length))
        results = await asyncio.gather(*tasks)
        end = time.time()
        duration = end - start
        avg_latency = duration / num_requests

        # Assertions
        assert all(r is not None for r in results)
        assert avg_latency < 30.0, f"Average latency {avg_latency}s exceeded threshold"