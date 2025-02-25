"""Connections to the OPIK server."""

import opik

from mirror_eval.core import config


def get_client(api_key: str | None = None, workspace: str | None = None) -> opik.Opik:
    """Gets the OPIK client.

    Args:
        api_key: The OPIK API key, if not provided taken from the OPIK_API_KEY
            environment variable.
        workspace: The OPIK workspace, if not provided taken from the
            OPIK_WORKSPACE environment variable.

    Returns:
          The OPIK client.
    """
    settings = config.get_settings()
    api_key = api_key or settings.OPIK_API_KEY
    workspace = workspace or settings.OPIK_WORKSPACE

    if not api_key or not workspace:
        msg = "Opik API key and workspace must be set."
        raise OSError(msg)

    opik.configure(api_key, workspace, force=True)
    return opik.Opik()
