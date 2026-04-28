
import pytest
import time
from unittest.mock import MagicMock, patch

# Import ONLY from 'agent'
from agent import SecurityManager

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def security_manager_instance():
    """Create SecurityManager with mocked Config.AGENT_ENCRYPTION_KEY."""
    with patch("agent.Config.AGENT_ENCRYPTION_KEY", new=None):
        instance = SecurityManager()
    return instance

# ── Unit Tests ──────────────────────────────────────────────────────

def test_unit_encrypt_data_happy_path():
    """Test SecurityManager.encrypt_data returns encrypted bytes for valid input."""
    # Patch Config.AGENT_ENCRYPTION_KEY to a known value
    key = b'0' * 32  # Fernet key must be 32 url-safe base64-encoded bytes
    import base64
    fernet_key = base64.urlsafe_b64encode(key)
    with patch("agent.Config.AGENT_ENCRYPTION_KEY", new=fernet_key):
        manager = SecurityManager()
        sample = "Sensitive meeting transcript"
        result = manager.encrypt_data(sample)
    assert result is not None
    assert isinstance(result, bytes)
    # Should not be equal to the plain utf-8 bytes
    assert result != sample.encode("utf-8")

def test_unit_encrypt_data_no_key_fallback():
    """Test SecurityManager.encrypt_data falls back to plain bytes if encryption key is missing."""
    with patch("agent.Config.AGENT_ENCRYPTION_KEY", new=None):
        manager = SecurityManager()
        sample = "Sensitive meeting transcript"
        result = manager.encrypt_data(sample)
    assert result is not None
    assert isinstance(result, bytes)
    # With no key, fallback is plain utf-8 bytes
    assert result == sample.encode("utf-8")