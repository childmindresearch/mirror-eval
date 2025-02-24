"""Tests for the opik.service module."""

import pytest

from mirror_eval.opik import service


def test_get_client_missing_arg() -> None:
    """Tests get_client with a missing argument results in OSError."""
    with pytest.raises(OSError, match="Opik API key and workspace must be set."):
        service.get_client(None, None)
