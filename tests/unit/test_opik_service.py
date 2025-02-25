"""Tests for the opik.service module."""


import pytest

from mirror_eval.core import config
from mirror_eval.opik import service


def test_get_client_missing_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests get_client with a missing argument results in OSError."""
    # Patch the get_settings function to return blank settings
    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: config.Settings(OPIK_API_KEY=None, OPIK_WORKSPACE=None),
    )

    with pytest.raises(OSError, match="Opik API key and workspace must be set."):
        service.get_client(None, None)
