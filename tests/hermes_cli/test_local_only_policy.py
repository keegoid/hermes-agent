"""Regression tests for the opt-in local-only Hermes inference fence."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hermes_cli.local_only_policy import (
    LocalOnlyViolation,
    enforce_local_provider_request,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8000/v1",
        "http://127.42.0.9:11434/v1",
        "http://localhost:8000/v1",
        "http://lab.localhost:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_local_only_allows_loopback_endpoints(monkeypatch, endpoint):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")

    enforce_local_provider_request(
        provider="custom",
        base_url=endpoint,
        surface="test",
    )


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("openai-codex", "https://chatgpt.com/backend-api/codex"),
        ("anthropic", "https://api.anthropic.com"),
        ("custom", "https://models.example.com/v1"),
        ("custom", "http://192.168.1.9:8000/v1"),
        ("custom", ""),
        ("moa", "moa://local"),
        ("custom:cloud-entry", "http://127.0.0.1:8000/v1"),
        ("actual", "http://127.0.0.1:8000/v1"),
        ("azure-foundry", "https://example.services.ai.azure.com/models"),
        ("gemini", "https://generativelanguage.googleapis.com"),
        ("ollama-cloud", "https://ollama.com/v1"),
        ("vertex", "https://us-central1-aiplatform.googleapis.com"),
    ],
)
def test_local_only_rejects_cloud_lan_and_missing_endpoints(
    monkeypatch,
    provider,
    endpoint,
):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "true")

    with pytest.raises(LocalOnlyViolation):
        enforce_local_provider_request(
            provider=provider,
            base_url=endpoint,
            surface="test",
        )


def test_policy_is_opt_in(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_ONLY", raising=False)

    enforce_local_provider_request(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        surface="test",
    )


def test_actual_client_transport_probe_is_opt_in(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_ONLY", raising=False)
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_actual_auxiliary_api_mode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("actual transport was probed")
        ),
    )

    auxiliary_client._enforce_local_auxiliary_client(
        SimpleNamespace(base_url="https://api.anthropic.com"),
        provider="anthropic",
        fallback_base_url=None,
        configured_api_mode=None,
        surface="test",
    )


@pytest.mark.parametrize(
    "api_mode",
    ["anthropic_messages", "bedrock_converse", "codex_app_server", "codex_responses"],
)
def test_local_only_rejects_non_chat_transports(monkeypatch, api_mode):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")

    with pytest.raises(LocalOnlyViolation):
        enforce_local_provider_request(
            provider="custom",
            base_url="http://127.0.0.1:8000/v1",
            surface="test",
            api_mode=api_mode,
        )


def test_local_only_rejects_external_model_command(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")

    with pytest.raises(LocalOnlyViolation):
        enforce_local_provider_request(
            provider="custom",
            base_url="http://127.0.0.1:8000/v1",
            surface="test",
            external_command="claude",
        )


def test_main_resolution_blocks_cloud_before_auth(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "resolve_codex_runtime_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("cloud auth was touched")),
    )

    with pytest.raises(LocalOnlyViolation):
        runtime_provider.resolve_runtime_provider(
            requested="openai-codex",
            explicit_base_url="https://chatgpt.com/backend-api/codex",
        )


def test_aux_resolution_blocks_cloud_before_client_build(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_build_codex_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud client was built")
        ),
    )

    with pytest.raises(LocalOnlyViolation):
        auxiliary_client.resolve_provider_client(
            "openai-codex",
            model="gpt-5.6-sol",
            explicit_base_url="https://chatgpt.com/backend-api/codex",
        )


def test_direct_agent_initialization_blocks_cloud(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_init import init_agent

    with pytest.raises(LocalOnlyViolation):
        init_agent(
            SimpleNamespace(),
            provider="anthropic",
            base_url="https://api.anthropic.com",
        )


def test_original_provider_identity_cannot_hide_behind_custom(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_init import init_agent

    with pytest.raises(LocalOnlyViolation, match="anthropic"):
        init_agent(
            SimpleNamespace(),
            provider="custom",
            requested_provider="anthropic",
            base_url="http://127.0.0.1:8000/v1",
        )


@pytest.mark.parametrize(
    "agent_kwargs",
    [
        {"acp_command": "claude"},
        {"api_mode": "codex_app_server"},
    ],
)
def test_direct_agent_initialization_blocks_external_transports(
    monkeypatch,
    agent_kwargs,
):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_init import init_agent

    with pytest.raises(LocalOnlyViolation):
        init_agent(
            SimpleNamespace(),
            provider="custom",
            base_url="http://127.0.0.1:8000/v1",
            **agent_kwargs,
        )


def test_aux_resolution_blocks_non_chat_transport(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    with pytest.raises(LocalOnlyViolation):
        auxiliary_client.resolve_provider_client(
            "custom",
            explicit_base_url="http://127.0.0.1:8000/v1",
            api_mode="anthropic_messages",
        )


def test_direct_model_switch_blocks_cloud_before_mutation(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_runtime_helpers import switch_model

    agent = SimpleNamespace(model="local", provider="custom")
    with pytest.raises(LocalOnlyViolation):
        switch_model(
            agent,
            "claude-fable-5",
            "anthropic",
            base_url="https://api.anthropic.com",
        )
    assert agent.model == "local"
    assert agent.provider == "custom"


def test_primary_client_creation_revalidates_actual_route(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_runtime_helpers import create_openai_client

    with pytest.raises(LocalOnlyViolation):
        create_openai_client(
            SimpleNamespace(provider="custom"),
            {"base_url": "https://models.example.com/v1"},
            reason="test",
            shared=False,
        )


def test_primary_client_creation_preserves_requested_provider(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_runtime_helpers import create_openai_client

    with pytest.raises(LocalOnlyViolation, match="anthropic"):
        create_openai_client(
            SimpleNamespace(
                provider="custom",
                requested_provider="anthropic",
            ),
            {
                "api_key": "local-key",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            reason="test",
            shared=False,
        )


def test_primary_client_creation_rejects_actual_non_chat_mode(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_runtime_helpers import create_openai_client

    with pytest.raises(LocalOnlyViolation, match="anthropic_messages"):
        create_openai_client(
            SimpleNamespace(
                provider="custom",
                api_mode="anthropic_messages",
            ),
            {
                "api_key": "local-key",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            reason="test",
            shared=False,
        )


def test_local_transport_never_uses_environment_proxy(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    from agent.process_bootstrap import _get_proxy_for_base_url

    assert _get_proxy_for_base_url("http://127.0.0.1:8000/v1") is None
    with pytest.raises(LocalOnlyViolation):
        _get_proxy_for_base_url("https://models.example.com/v1")


def test_local_transport_build_failure_is_fatal(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    import httpx
    from agent.process_bootstrap import build_keepalive_http_client

    def _fail_limits(**_kwargs):
        raise RuntimeError("transport build failed")

    monkeypatch.setattr(httpx, "Limits", _fail_limits)
    with pytest.raises(RuntimeError, match="transport build failed"):
        build_keepalive_http_client("http://127.0.0.1:8000/v1")


def test_primary_transport_build_failure_is_fatal(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.agent_runtime_helpers import create_openai_client

    agent = SimpleNamespace(
        provider="custom",
        _build_keepalive_http_client=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(LocalOnlyViolation, match="proxy-free primary"):
        create_openai_client(
            agent,
            {
                "api_key": "local-key",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            reason="test",
            shared=False,
        )


def test_payment_fallback_chain_is_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_get_provider_chain",
        lambda: (_ for _ in ()).throw(AssertionError("fallback chain was inspected")),
    )

    assert auxiliary_client._try_payment_fallback("custom") == (None, None, "")


def test_aux_fallback_revalidates_actual_route(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent.auxiliary_client import _call_fallback_candidate_sync

    client = SimpleNamespace(base_url="https://api.anthropic.com")
    with pytest.raises(LocalOnlyViolation):
        _call_fallback_candidate_sync(
            client,
            "claude-haiku",
            "configured-fallback",
            task="compression",
            messages=[],
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=30,
            effective_extra_body={},
            reasoning_config=None,
        )


def test_aux_fallback_blocks_non_chat_destination_before_replanning(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    client = SimpleNamespace(
        base_url="http://127.0.0.1:8000/v1",
        _hermes_fallback_destination=auxiliary_client._FallbackDestination(
            "custom",
            "http://127.0.0.1:8000/v1",
            "anthropic_messages",
            "local-model",
        ),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_replan_synchronous_cache_sections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache replanning ran")
        ),
    )

    with pytest.raises(LocalOnlyViolation, match="anthropic_messages"):
        auxiliary_client._call_fallback_candidate_sync(
            client,
            "local-model",
            "configured-fallback",
            task="compression",
            messages=[],
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=30,
            effective_extra_body={},
            reasoning_config=None,
        )


def test_aux_fallback_preserves_cloud_provider_identity(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    client = SimpleNamespace(
        base_url="http://127.0.0.1:8000/v1",
        _hermes_fallback_destination=auxiliary_client._FallbackDestination(
            "anthropic",
            "http://127.0.0.1:8000/v1",
            "chat_completions",
            "local-model",
        ),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_replan_synchronous_cache_sections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache replanning ran")
        ),
    )

    with pytest.raises(LocalOnlyViolation, match="anthropic"):
        auxiliary_client._call_fallback_candidate_sync(
            client,
            "local-model",
            "configured-fallback",
            task="compression",
            messages=[],
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=30,
            effective_extra_body={},
            reasoning_config=None,
        )


def test_fallback_resolution_blocks_before_cloud_key_lookup(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_fallback_entry_api_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud fallback key was inspected")
        ),
    )

    with pytest.raises(LocalOnlyViolation, match="anthropic"):
        auxiliary_client._resolve_fallback_entry(
            {
                "provider": "anthropic",
                "model": "claude-haiku",
                "base_url": "https://api.anthropic.com",
                "api_key_env": "ANTHROPIC_API_KEY",
            }
        )


def test_primary_fallback_blocks_before_cloud_auth_lookup(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import chat_completion_helpers
    from hermes_cli import fallback_config

    monkeypatch.setattr(
        chat_completion_helpers,
        "_fallback_entry_unavailable_without_network",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud fallback auth state was inspected")
        ),
    )
    monkeypatch.setattr(
        fallback_config,
        "resolve_entry_api_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud fallback key was inspected")
        ),
    )
    agent = SimpleNamespace(
        _fallback_chain=[
            {
                "provider": "anthropic",
                "model": "claude-haiku",
                "base_url": "https://api.anthropic.com",
                "api_key_env": "ANTHROPIC_API_KEY",
            }
        ],
        _fallback_index=0,
        _unavailable_fallback_keys=set(),
        _try_activate_fallback=lambda _reason=None: False,
    )

    assert chat_completion_helpers.try_activate_fallback(agent) is False
    assert agent._fallback_index == 1


def test_missing_custom_route_never_scans_api_key_providers(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_try_custom_endpoint",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_api_key_provider",
        lambda: (_ for _ in ()).throw(
            AssertionError("cloud API-key providers were scanned")
        ),
    )

    assert auxiliary_client.resolve_provider_client(
        "custom",
        model="local-model",
        main_runtime={
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "",
        },
    ) == (None, None)


def test_named_custom_route_blocks_before_key_command(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client, command_token_source
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "_get_named_custom_provider",
        lambda _provider: {
            "name": "local",
            "model": "local-model",
            "base_url": "https://models.example.com/v1",
            "key_cmd": "cloud-key-helper",
        },
    )
    monkeypatch.setattr(
        command_token_source,
        "build_command_token_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloud key command was executed")
        ),
    )

    with pytest.raises(LocalOnlyViolation):
        auxiliary_client.resolve_provider_client(
            "local",
            model="local-model",
            explicit_base_url="http://127.0.0.1:8000/v1",
        )


def test_provider_discovery_chain_is_empty(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    assert auxiliary_client._get_provider_chain() == []


def test_auto_route_stops_with_three_value_result(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "resolve_provider_client",
        lambda *_args, **_kwargs: (None, None),
    )

    assert auxiliary_client._resolve_auto_route(
        main_runtime={
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_mode": "chat_completions",
        }
    ) == (None, None, "")


def test_anthropic_url_autowrap_is_blocked_before_native_client_build(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    from agent import anthropic_adapter, auxiliary_client

    monkeypatch.setattr(
        anthropic_adapter,
        "build_anthropic_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native Anthropic client was built")
        ),
    )

    with pytest.raises(LocalOnlyViolation, match="anthropic_messages"):
        auxiliary_client._maybe_wrap_anthropic(
            SimpleNamespace(),
            "local-model",
            "local-key",
            "http://127.0.0.1:8000/anthropic",
        )


def test_strict_vision_cloud_backend_is_skipped_before_auth(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_try_openrouter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("vision cloud auth was touched")
        ),
    )

    assert auxiliary_client._resolve_strict_vision_backend("openrouter") == (
        None,
        None,
    )


class _ForbiddenCompletions:
    def __init__(self):
        self.called = False

    def create(self, **_kwargs):
        self.called = True
        raise AssertionError("cloud dispatch was attempted")


class _ForbiddenAsyncCompletions:
    def __init__(self):
        self.called = False

    async def create(self, **_kwargs):
        self.called = True
        raise AssertionError("cloud dispatch was attempted")


def _cloud_client(completions):
    return SimpleNamespace(
        base_url="https://models.example.com/v1",
        chat=SimpleNamespace(completions=completions),
    )


def _local_resolution(*_args, **_kwargs):
    return (
        "custom",
        "local-model",
        "http://127.0.0.1:8000/v1",
        "local-key",
        "chat_completions",
    )


def _local_resolution_without_mode(*_args, **_kwargs):
    return (
        "custom",
        "local-model",
        "http://127.0.0.1:8000/anthropic",
        "local-key",
        None,
    )


def _cloud_identity_local_resolution(*_args, **_kwargs):
    return (
        "anthropic",
        "local-model",
        "http://127.0.0.1:8000/v1",
        "local-key",
        "chat_completions",
    )


def _anthropic_loopback_client(*, async_mode=False):
    from agent import auxiliary_client

    sync_client = auxiliary_client.AnthropicAuxiliaryClient(
        SimpleNamespace(close=lambda: None),
        "local-model",
        "local-key",
        "http://127.0.0.1:8000/anthropic",
    )
    if async_mode:
        return auxiliary_client.AsyncAnthropicAuxiliaryClient(sync_client)
    return sync_client


def test_sync_dispatch_revalidates_cached_client(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    completions = _ForbiddenCompletions()
    client = _cloud_client(completions)
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        _local_resolution,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "local-model"),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_relay_sync_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relay dispatch was attempted")
        ),
    )

    with pytest.raises(LocalOnlyViolation):
        auxiliary_client.call_llm(
            messages=[{"role": "user", "content": "test"}],
        )
    assert completions.called is False


def test_sync_dispatch_preserves_cached_cloud_provider_identity(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    completions = _ForbiddenCompletions()
    client = SimpleNamespace(
        base_url="http://127.0.0.1:8000/v1",
        chat=SimpleNamespace(completions=completions),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        _cloud_identity_local_resolution,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "local-model"),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_relay_sync_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relay dispatch was attempted")
        ),
    )

    with pytest.raises(LocalOnlyViolation, match="anthropic"):
        auxiliary_client.call_llm(
            messages=[{"role": "user", "content": "test"}],
        )
    assert completions.called is False


def test_sync_dispatch_rejects_auto_wrapped_anthropic_loopback(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        _local_resolution_without_mode,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (_anthropic_loopback_client(), "local-model"),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_relay_sync_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relay dispatch was attempted")
        ),
    )

    with pytest.raises(LocalOnlyViolation, match="anthropic_messages"):
        auxiliary_client.call_llm(
            messages=[{"role": "user", "content": "test"}],
        )


def test_async_dispatch_revalidates_cached_client(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    completions = _ForbiddenAsyncCompletions()
    client = _cloud_client(completions)
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        _local_resolution,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "local-model"),
    )

    async def _forbidden_relay(*_args, **_kwargs):
        raise AssertionError("relay dispatch was attempted")

    monkeypatch.setattr(
        auxiliary_client,
        "_relay_async_completion",
        _forbidden_relay,
    )

    async def _run():
        with pytest.raises(LocalOnlyViolation):
            await auxiliary_client.async_call_llm(
                messages=[{"role": "user", "content": "test"}],
            )

    asyncio.run(_run())
    assert completions.called is False


def test_async_dispatch_rejects_auto_wrapped_anthropic_loopback(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        _local_resolution_without_mode,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (
            _anthropic_loopback_client(async_mode=True),
            "local-model",
        ),
    )

    async def _forbidden_relay(*_args, **_kwargs):
        raise AssertionError("relay dispatch was attempted")

    monkeypatch.setattr(
        auxiliary_client,
        "_relay_async_completion",
        _forbidden_relay,
    )

    async def _run():
        with pytest.raises(LocalOnlyViolation, match="anthropic_messages"):
            await auxiliary_client.async_call_llm(
                messages=[{"role": "user", "content": "test"}],
            )

    asyncio.run(_run())


def _retry_kwargs():
    return {
        "task": None,
        "resolved_provider": "custom",
        "resolved_model": "local-model",
        "resolved_base_url": "http://127.0.0.1:8000/v1",
        "resolved_api_key": "local-key",
        "resolved_api_mode": "chat_completions",
        "final_model": "local-model",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": None,
        "max_tokens": None,
        "tools": None,
        "effective_timeout": 30,
        "effective_extra_body": {},
        "reasoning_config": None,
    }


def test_sync_retry_revalidates_rebuilt_client(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    completions = _ForbiddenCompletions()
    client = _cloud_client(completions)
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "local-model"),
    )

    with pytest.raises(LocalOnlyViolation):
        auxiliary_client._retry_same_provider_sync(
            main_runtime=None,
            **_retry_kwargs(),
        )
    assert completions.called is False


def test_async_retry_revalidates_rebuilt_client(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_ONLY", "1")
    from agent import auxiliary_client

    completions = _ForbiddenAsyncCompletions()
    client = _cloud_client(completions)
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "local-model"),
    )

    async def _run():
        with pytest.raises(LocalOnlyViolation):
            await auxiliary_client._retry_same_provider_async(
                **_retry_kwargs(),
            )

    asyncio.run(_run())
    assert completions.called is False
