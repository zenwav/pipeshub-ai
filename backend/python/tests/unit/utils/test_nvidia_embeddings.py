"""Unit tests for NvidiaEmbeddings and its aimodels factory wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from app.config.constants.ai_models import NVIDIA_NIM_BASE_URL
from app.services.llm_gateway.embedding_gateway import EmbeddingRateLimited
from app.utils.aimodels import (
    EmbeddingProvider,
    get_embedding_model,
    parse_embedding_dimensions,
)
from app.utils.nvidia_embeddings import (
    DEFAULT_HOSTED_NVIDIA_RPM,
    DEFAULT_NVIDIA_INTERACTIVE_RESERVE_RPM,
    DEFAULT_RATE_LIMIT_DEFER_S,
    NvidiaEmbeddings,
    get_nvidia_endpoint_pacer,
    is_nvidia_hosted_endpoint,
    nvidia_pacer_limits,
)
from app.utils.nvidia_pacer import Lane, PacerBusy, PacerLimits, clear_nvidia_pacers


def _resp(vectors: list[tuple[int, list[float]]]) -> MagicMock:
    items = []
    for idx, vec in vectors:
        item = MagicMock()
        item.index = idx
        item.embedding = vec
        items.append(item)
    response = MagicMock()
    response.data = items
    return response


def _rate_limit_error(headers: dict[str, str] | None = None) -> openai.RateLimitError:
    req = httpx.Request("POST", f"{NVIDIA_NIM_BASE_URL}/embeddings")
    return openai.RateLimitError(
        "Too Many Requests",
        response=httpx.Response(429, request=req, headers=headers or {}),
        body={"status": 429, "title": "Too Many Requests"},
    )


def _with_fake_pacer(emb: NvidiaEmbeddings) -> MagicMock:
    pacer = MagicMock()
    pacer.acquire = AsyncMock()
    emb._pacer = pacer
    return pacer


@pytest.fixture(autouse=True)
def _reset_pacers(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_nvidia_pacers()
    monkeypatch.setenv("NVIDIA_NIM_RPM", "0")
    yield
    clear_nvidia_pacers()


# ---------------------------------------------------------------------------
# Dimension parser
# ---------------------------------------------------------------------------


def test_parse_embedding_dimensions_handles_blank_zero_negative_and_strings() -> None:
    assert parse_embedding_dimensions(None) is None
    assert parse_embedding_dimensions("") is None
    assert parse_embedding_dimensions("   ") is None
    assert parse_embedding_dimensions(0) is None
    assert parse_embedding_dimensions(-5) is None
    assert parse_embedding_dimensions(True) is None
    assert parse_embedding_dimensions(False) is None
    assert parse_embedding_dimensions("abc") is None
    assert parse_embedding_dimensions("2048") == 2048
    assert parse_embedding_dimensions(1024) == 1024


# ---------------------------------------------------------------------------
# Hosted endpoint detection and pacer limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("", True),
        ("   ", True),
        ("https://integrate.api.nvidia.com/v1", True),
        ("https://integrate.api.nvidia.com/v1/", True),
        ("https://ai.api.nvidia.com/v1", True),
        ("https://nim.internal.example/v1", False),
        ("https://evil-nvidia.com/v1", False),
        ("http://localhost:8000/v1", False),
    ],
)
def test_is_nvidia_hosted_endpoint(url: str, expected: bool) -> None:
    assert is_nvidia_hosted_endpoint(url) is expected


def test_pacer_limits_hosted_vs_self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_NIM_RPM", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_INTERACTIVE_RESERVE_RPM", raising=False)
    hosted = nvidia_pacer_limits("https://integrate.api.nvidia.com/v1")
    assert hosted == PacerLimits(
        DEFAULT_HOSTED_NVIDIA_RPM, DEFAULT_HOSTED_NVIDIA_RPM - DEFAULT_NVIDIA_INTERACTIVE_RESERVE_RPM
    )
    assert hosted.total < 40
    assert nvidia_pacer_limits("https://nim.internal.example/v1").enabled is False


def test_pacer_limits_follow_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_NIM_RPM", "20")
    monkeypatch.setenv("NVIDIA_NIM_INTERACTIVE_RESERVE_RPM", "5")
    assert nvidia_pacer_limits("https://integrate.api.nvidia.com/v1") == PacerLimits(20, 15)
    monkeypatch.setenv("NVIDIA_NIM_RPM", "not-a-number")
    assert nvidia_pacer_limits("https://integrate.api.nvidia.com/v1").total == DEFAULT_HOSTED_NVIDIA_RPM
    monkeypatch.setenv("NVIDIA_NIM_RPM", "0")
    assert nvidia_pacer_limits("https://integrate.api.nvidia.com/v1").enabled is False


def test_text_and_image_callers_of_one_key_share_a_pacer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_NIM_RPM", raising=False)
    hosted = get_nvidia_endpoint_pacer("https://integrate.api.nvidia.com/v1/", "nvapi-key-1")
    assert get_nvidia_endpoint_pacer("https://integrate.api.nvidia.com/v1", "nvapi-key-1") is hosted
    assert get_nvidia_endpoint_pacer("https://integrate.api.nvidia.com/v1", "nvapi-key-2") is not hosted


# ---------------------------------------------------------------------------
# NvidiaEmbeddings text embedding behaviour
# ---------------------------------------------------------------------------


def test_embed_documents_sends_passage_and_batches_in_index_order() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test", batch_size=2)
    emb.client = MagicMock()
    emb.client.create.side_effect = [
        _resp([(1, [2.0, 2.0]), (0, [1.0, 1.0])]),
        _resp([(0, [3.0, 3.0])]),
    ]
    out = emb.embed_documents(["a", "b", "c"])
    assert out == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert emb.client.create.call_count == 2
    assert emb.client.create.call_args_list[0].kwargs == {
        "model": "nvidia/nemotron-3.5-embed-8b",
        "input": ["a", "b"],
        "encoding_format": "float",
        "extra_body": {"input_type": "passage", "truncate": "END"},
    }
    assert emb.client.create.call_args_list[1].kwargs["input"] == ["c"]


def test_embed_documents_empty_list_makes_no_http_call() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    assert emb.embed_documents([]) == []
    emb.client.create.assert_not_called()


async def test_aembed_documents_empty_and_batched_with_dimensions() -> None:
    emb = NvidiaEmbeddings(
        model="nvidia/nemotron-3.5-embed-8b",
        api_key="nvapi-test",
        dimensions=1024,
        batch_size=2,
    )
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(return_value=_resp([(0, [0.5] * 1024)]))
    assert await emb.aembed_documents([]) == []
    out = await emb.aembed_documents(["slide 1"])
    assert len(out) == 1 and len(out[0]) == 1024
    kwargs = emb.async_client.create.call_args.kwargs
    assert kwargs["dimensions"] == 1024
    assert kwargs["encoding_format"] == "float"
    assert kwargs["extra_body"] == {"input_type": "passage", "truncate": "END"}


def test_embed_query_sends_query_input_type() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    emb.client.create.return_value = _resp([(0, [0.1, 0.2])])
    assert emb.embed_query("what is backpropagation?") == [0.1, 0.2]
    assert emb.client.create.call_args.kwargs == {
        "model": "nvidia/nemotron-3.5-embed-8b",
        "input": ["what is backpropagation?"],
        "encoding_format": "float",
        "extra_body": {"input_type": "query", "truncate": "END"},
    }


async def test_aembed_query_sends_query_input_type_with_dimensions() -> None:
    emb = NvidiaEmbeddings(
        model="nvidia/nemotron-3.5-embed-8b",
        api_key="nvapi-test",
        dimensions=512,
    )
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(return_value=_resp([(0, [0.3] * 512)]))
    vec = await emb.aembed_query("gradient descent")
    assert len(vec) == 512
    assert emb.async_client.create.call_args.kwargs["extra_body"] == {
        "input_type": "query",
        "truncate": "END",
    }
    assert emb.async_client.create.call_args.kwargs["dimensions"] == 512


def test_truncated_or_empty_response_raises_value_error() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    emb.client.create.return_value = _resp([(0, [1.0])])
    with pytest.raises(ValueError, match="expected 2 embeddings, got 1"):
        emb.embed_documents(["a", "b"])
    emb.client.create.return_value = _resp([(0, [])])
    with pytest.raises(ValueError, match="empty embedding"):
        emb.embed_documents(["a"])


def test_repeated_or_out_of_range_indexes_are_refused() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    emb.client.create.return_value = _resp([(0, [1.0]), (0, [2.0])])
    with pytest.raises(ValueError, match="indexes that do not match"):
        emb.embed_documents(["a", "b"])
    emb.client.create.return_value = _resp([(0, [1.0]), (5, [2.0])])
    with pytest.raises(ValueError, match="indexes that do not match"):
        emb.embed_documents(["a", "b"])


async def test_headerless_429_on_documents_slows_the_key_and_marks_for_deferral() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    pacer = _with_fake_pacer(emb)
    err = _rate_limit_error()
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(side_effect=err)
    with pytest.raises(EmbeddingRateLimited) as raised:
        await emb.aembed_documents(["slide 1"])
    assert raised.value.retry_after == DEFAULT_RATE_LIMIT_DEFER_S
    assert raised.value.__cause__ is err
    pacer.acquire.assert_awaited_once_with(Lane.BULK)
    pacer.penalize.assert_called_once_with(15.0)
    # NVIDIA's answer is left as it came.
    assert "retry-after" not in err.response.headers


async def test_429_with_retry_after_is_left_to_the_callers_policy() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    pacer = _with_fake_pacer(emb)
    err = _rate_limit_error({"retry-after": "30"})
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(side_effect=err)
    with pytest.raises(openai.RateLimitError):
        await emb.aembed_documents(["slide 1"])
    pacer.penalize.assert_called_once_with(30.0)


async def test_headerless_429_on_a_search_keeps_the_callers_retries() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    pacer = _with_fake_pacer(emb)
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(side_effect=_rate_limit_error())
    with pytest.raises(openai.RateLimitError):
        await emb.aembed_query("what is a branch predictor?")
    pacer.acquire.assert_awaited_once_with(Lane.INTERACTIVE)
    pacer.penalize.assert_called_once_with(15.0)


async def test_quota_exhausted_429_neither_slows_the_key_nor_defers() -> None:
    emb = NvidiaEmbeddings(api_key="nvapi-test", model="nvidia/nemotron-3.5-embed-8b")
    pacer = _with_fake_pacer(emb)
    req = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings")
    quota_err = openai.RateLimitError(
        "Out of credit",
        response=httpx.Response(429, request=req),
        body={"error": {"message": "Out of credit", "code": "insufficient_quota"}},
    )
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(side_effect=quota_err)
    with pytest.raises(openai.RateLimitError):
        await emb.aembed_documents(["slide 1"])
    pacer.penalize.assert_not_called()


async def test_a_saturated_pacer_hands_the_batch_back_without_sending() -> None:
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    pacer = _with_fake_pacer(emb)
    pacer.acquire = AsyncMock(side_effect=PacerBusy(80.0))
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock()
    with pytest.raises(EmbeddingRateLimited) as raised:
        await emb.aembed_documents(["slide 1"])
    assert raised.value.retry_after == 80.0
    emb.async_client.create.assert_not_awaited()


def test_a_deferred_batch_is_not_retried_by_the_gateway() -> None:
    from app.services.llm_gateway.embedding_gateway import EmbeddingGateway
    from app.services.spend_guard.retry_policy import RetryBudget

    limited = EmbeddingRateLimited("deferring", retry_after=60.0)
    assert EmbeddingGateway(retry_budget=RetryBudget()).retry_kind(limited) is None


def test_gateway_identity_and_endpoint_path_guard() -> None:
    from app.services.llm_gateway.fleet_breaker import credential_of
    from app.services.llm_gateway.gateway import provider_key
    from app.utils.openai_hosts import endpoint_path_problem

    hosted = NvidiaEmbeddings(api_key="nvapi-key-a", model="nvidia/nemotron-3.5-embed-8b")
    self_hosted = NvidiaEmbeddings(
        api_key="nvapi-key-b",
        model="nvidia/nemotron-3.5-embed-8b",
        base_url="http://nim.local:8000/v1/",
    )
    assert provider_key(hosted) == f"NvidiaEmbeddings:nvidia/nemotron-3.5-embed-8b@{NVIDIA_NIM_BASE_URL}"
    assert provider_key(self_hosted) == "NvidiaEmbeddings:nvidia/nemotron-3.5-embed-8b@http://nim.local:8000/v1"
    assert credential_of(hosted) != credential_of(self_hosted)

    problem = endpoint_path_problem("nvidia", "https://integrate.api.nvidia.com/v1/embeddings")
    assert problem is not None
    assert '"/embeddings"' in problem


# ---------------------------------------------------------------------------
# Factory wiring (aimodels.py)
# ---------------------------------------------------------------------------


def test_factory_builds_nvidia_default_and_custom_endpoint() -> None:
    default_emb = get_embedding_model(
        EmbeddingProvider.NVIDIA.value,
        {
            "configuration": {
                "apiKey": "nvapi-k",
                "model": "nvidia/nemotron-3.5-embed-8b",
                "endpoint": "   ",
            },
            "isDefault": True,
        },
    )
    assert isinstance(default_emb, NvidiaEmbeddings)
    assert default_emb.nvidia_api_base == NVIDIA_NIM_BASE_URL
    assert default_emb.max_retries == 0

    custom_emb = get_embedding_model(
        EmbeddingProvider.NVIDIA.value,
        {
            "configuration": {
                "apiKey": "nvapi-k",
                "model": "nvidia/nemotron-3.5-embed-8b",
                "endpoint": "https://nim.example.com/v1/",
                "dimensions": "2048",
            },
            "isDefault": True,
        },
    )
    assert isinstance(custom_emb, NvidiaEmbeddings)
    assert custom_emb.nvidia_api_base == "https://nim.example.com/v1"
    assert custom_emb.dimensions == 2048


def test_factory_blocks_metadata_endpoint() -> None:
    with pytest.raises(ValueError, match="link-local or cloud metadata"):
        get_embedding_model(
            EmbeddingProvider.NVIDIA.value,
            {
                "configuration": {
                    "apiKey": "nvapi-k",
                    "model": "nvidia/nemotron-3.5-embed-8b",
                    "endpoint": "http://169.254.169.254/v1",
                },
                "isDefault": True,
            },
        )


def test_registry_and_discovery_expose_nvidia_nim_embedding() -> None:
    import app.config.ai_models.providers  # noqa: F401
    from app.config.ai_models.registry import ai_model_registry
    from app.services.ai_models.discovery.registry import discovery_registry

    provider = ai_model_registry.get_provider("nvidia")
    assert provider is not None
    assert provider["name"] == "NVIDIA NIM"
    assert provider["capabilities"] == ["embedding"]
    assert provider["iconPath"] == "/icons/ai-models/nvidia-color.svg"
    fields = {f["name"]: f for f in provider["fields"]["embedding"]}
    assert fields["apiKey"]["required"] is True
    assert fields["model"]["defaultValue"] == "nvidia/nemotron-3.5-embed-8b"
    assert fields["endpoint"]["required"] is False
    assert fields["isMultimodal"]["defaultValue"] is False
    assert "dimensions" in fields and "modelFriendlyName" in fields

    strategy = discovery_registry.get("nvidia")
    assert strategy is not None
    assert strategy.mode == "static"
