"""Fail-closed inference policy for Keegoid's local-model Hermes lane.

The policy is opt-in through ``HERMES_LOCAL_ONLY=1`` so upstream behavior and
tests remain unchanged.  When enabled, every model-provider resolution must
name a local/custom provider and an HTTP(S) loopback endpoint.  Stored cloud
credentials remain available to Codex and Claude Code, but Hermes cannot use
them—even through provider overrides or ``--ignore-user-config``.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Optional
from urllib.parse import urlsplit


_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOCAL_PROVIDER_NAMES = {
    "auto",
    "custom",
    "llamacpp",
    "lm-studio",
    "lm_studio",
    "lmstudio",
    "local",
    "main",
    "ollama",
    "vllm",
}


class LocalOnlyViolation(RuntimeError):
    """Raised before Hermes can resolve credentials for a cloud model."""


def local_only_enabled() -> bool:
    """Return whether this process is fenced to loopback inference."""
    return os.getenv("HERMES_LOCAL_ONLY", "").strip().lower() in _TRUE_VALUES


def _is_loopback_endpoint(base_url: str) -> bool:
    """Accept HTTP(S) endpoints whose hostname is unambiguously loopback."""
    try:
        parsed = urlsplit(str(base_url or "").strip())
    except (TypeError, ValueError):
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def enforce_local_provider_request(
    *,
    provider: Optional[str],
    base_url: Optional[str],
    surface: str,
    api_mode: Optional[str] = None,
    external_command: Optional[str] = None,
) -> None:
    """Reject an inference route before it can inspect cloud auth or dispatch.

    ``auto`` is allowed only with an already-known loopback base URL.  That
    preserves local main-model inheritance while making an empty/misconfigured
    endpoint fail closed instead of entering Hermes' cloud fallback chain.
    """
    if not local_only_enabled():
        return

    command = str(external_command or "").strip()
    if command:
        raise LocalOnlyViolation(
            f"Hermes local-only policy blocked external model command "
            f"{command!r} on {surface}."
        )

    normalized_api_mode = str(api_mode or "chat_completions").strip().lower()
    if normalized_api_mode != "chat_completions":
        raise LocalOnlyViolation(
            f"Hermes local-only policy blocked API mode "
            f"{normalized_api_mode!r} on {surface}."
        )

    normalized_provider = str(provider or "auto").strip().lower()
    provider_is_local = normalized_provider in _LOCAL_PROVIDER_NAMES
    if not provider_is_local:
        raise LocalOnlyViolation(
            f"Hermes local-only policy blocked provider "
            f"{normalized_provider!r} on {surface}."
        )

    endpoint = str(base_url or "").strip()
    if not _is_loopback_endpoint(endpoint):
        shown = endpoint or "<unset>"
        raise LocalOnlyViolation(
            f"Hermes local-only policy requires a loopback inference endpoint "
            f"on {surface}; got {shown!r}."
        )
