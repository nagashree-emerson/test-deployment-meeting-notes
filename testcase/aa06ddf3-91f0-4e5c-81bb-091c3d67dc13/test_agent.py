
import pytest
from unittest.mock import patch
from agent import SecurityManager
import types

def test_securitymanager_encrypt_data_with_and_without_key(monkeypatch):
    """
    Verifies SecurityManager.encrypt_data encrypts data and can handle missing encryption key.
    """
    sample_text = "Sensitive meeting transcript data"
    # Patch Config.AGENT_ENCRYPTION_KEY to a valid Fernet key
    from cryptography.fernet import Fernet

    valid_key = Fernet.generate_key()
    monkeypatch.setattr("agent.Config.AGENT_ENCRYPTION_KEY", valid_key, raising=False)
    sm = SecurityManager()
    encrypted = sm.encrypt_data(sample_text)
    assert isinstance(encrypted, bytes)
    # Should not be the same as utf-8 encoded input
    assert encrypted != sample_text.encode("utf-8")

    # Patch Config.AGENT_ENCRYPTION_KEY to None (simulate missing key)
    monkeypatch.setattr("agent.Config.AGENT_ENCRYPTION_KEY", None, raising=False)
    sm_no_key = SecurityManager()
    output = sm_no_key.encrypt_data(sample_text)
    assert isinstance(output, bytes)
    # Should be equal to utf-8 encoded input (no encryption fallback)
    assert output == sample_text.encode("utf-8")