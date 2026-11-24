"""Unit tests for NvidiaMultimodalProvider: MIME accuracy, batching, fallback, and 429 deferral."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from PIL import Image

from app.config.constants.ai_models import NVIDIA_NIM_BASE_URL
from app.exceptions.non_retryable import NonRetryableError
from app.services.embeddings.multimodal.nvidia_provider import (
    NvidiaMultimodalProvider,
    _group_image_batches,
)
from app.services.llm_gateway.embedding_gateway import (
    EmbeddingGateway,
    EmbeddingRateLimited,
)
from app.services.llm_gateway.wire import SendAllowanceSpent
from app.services.spend_guard.retry_policy import RetryBudget
from app.utils.nvidia_pacer import Lane, PacerBusy, clear_nvidia_pacers


def _make_image_b64(fmt: str, mode: str = "RGB", size: tuple[int, int] = (4, 4)) -> str:
    buf = io.BytesIO()
    img = Image.new(mode, size, color=(12, 180, 40) if mode == "RGB" else (12, 180, 40, 255))
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(autouse=True)
def _reset_pacers(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_nvidia_pacers()
    monkeypatch.setenv("NVIDIA_NIM_RPM", "0")
    yield
    clear_nvidia_pacers()


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> list[tuple[str, dict]]:
    captured: list[tuple[str, dict]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((str(request.url), body))
        return handler(request, body)

    def make_client(_base_url: str, *, timeout: float = 60.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(transport_handler),
            timeout=timeout,
        )

    monkeypatch.setattr(
        "app.services.embeddings.multimodal.nvidia_provider.guarded_async_client",
        make_client,
    )
    gateway = EmbeddingGateway(retry_budget=RetryBudget(), max_attempts=1)
    monkeypatch.setattr(
        "app.services.embeddings.multimodal.nvidia_provider.get_embedding_gateway",
        lambda: gateway,
    )
    return captured


# ---------------------------------------------------------------------------
# Batch grouping helper
# ---------------------------------------------------------------------------


def test_group_image_batches_caps_count_and_payload_bytes() -> None:
    # 5 small images -> [4, 1]
    small = [(i, "x" * 100) for i in range(5)]
    batches = _group_image_batches(small, max_images=4, max_bytes=1000)
    assert [[i for i, _ in b] for b in batches] == [[0, 1, 2, 3], [4]]

    # Oversized images split by payload cap; a single giant image goes alone.
    sized = [(0, "a" * 600), (1, "b" * 600), (2, "c" * 2000), (3, "d" * 100)]
    batches = _group_image_batches(sized, max_images=4, max_bytes=1000)
    assert [[i for i, _ in b] for b in batches] == [[0], [1], [2], [3]]


# ---------------------------------------------------------------------------
# MIME detection, conversion, and payload format
# ---------------------------------------------------------------------------


async def test_png_and_jpeg_preserve_magic_byte_mime_even_when_prefix_mislabeled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_b64 = _make_image_b64("PNG")
    jpeg_b64 = _make_image_b64("JPEG")
    # Deliberately mislabel PNG with image/jpeg prefix to verify magic-byte check wins.
    mislabeled_png = f"data:image/jpeg;base64,{png_b64}"

    def handler(_req: httpx.Request, body: dict) -> httpx.Response:
        data = [{"index": i, "embedding": [float(i + 1), 0.5]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data})

    captured = _patch_client(monkeypatch, handler)
    provider = NvidiaMultimodalProvider(
        api_key="nvapi-k",
        model_name="nvidia/nemotron-3.5-embed-8b",
        base_url="https://integrate.api.nvidia.com/v1/",
    )
    results = await provider.embed_images([mislabeled_png, jpeg_b64])

    assert len(captured) == 1
    url, payload = captured[0]
    assert url == f"{NVIDIA_NIM_BASE_URL}/embeddings"
    assert payload["model"] == "nvidia/nemotron-3.5-embed-8b"
    assert payload["encoding_format"] == "float"
    assert payload["input_type"] == "passage"
    assert payload["modality"] == "image"
    assert payload["truncate"] == "END"
    assert "dimensions" not in payload
    assert payload["input"][0] == f"data:image/png;base64,{png_b64}"
    assert payload["input"][1] == f"data:image/jpeg;base64,{jpeg_b64}"
    assert [r.embedding for r in sorted(results, key=lambda r: r.index)] == [[1.0, 0.5], [2.0, 0.5]]


async def test_bmp_and_webp_convert_losslessly_to_rgb_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bmp_b64 = _make_image_b64("BMP")
    webp_b64 = _make_image_b64("WEBP", mode="RGBA")

    def handler(_req: httpx.Request, body: dict) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [0.2, 0.4]} for i in range(len(body["input"]))]},
        )

    captured = _patch_client(monkeypatch, handler)
    provider = NvidiaMultimodalProvider(
        api_key="nvapi-k",
        model_name="nvidia/nemotron-3.5-embed-8b",
        dimensions=1024,
    )
    results = await provider.embed_images([bmp_b64, webp_b64])
    assert all(r.embedding == [0.2, 0.4] for r in results)

    _, payload = captured[0]
    assert payload["dimensions"] == 1024
    for uri in payload["input"]:
        assert uri.startswith("data:image/png;base64,")
        raw = base64.b64decode(uri.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as decoded:
            assert decoded.format == "PNG"
            assert decoded.mode == "RGB"


async def test_corrupt_image_isolated_from_valid_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_b64 = _make_image_b64("PNG")
    jpeg_b64 = _make_image_b64("JPEG")

    def handler(_req: httpx.Request, body: dict) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [float(i + 1)]} for i in range(len(body["input"]))]},
        )

    _patch_client(monkeypatch, handler)
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    results = sorted(
        await provider.embed_images([png_b64, "not-valid-b64!!!", jpeg_b64]),
        key=lambda r: r.index,
    )
    assert results[0].embedding == [1.0]
    assert results[1].embedding is None and results[1].error == "invalid image data"
    assert results[2].embedding == [2.0]


async def test_multi_image_batch_400_falls_back_to_single_image_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_png = _make_image_b64("PNG")
    bad_jpeg = _make_image_b64("JPEG")

    def handler(_req: httpx.Request, body: dict) -> httpx.Response:
        # Fail the combined batch, then succeed for PNG alone and reject JPEG alone.
        if len(body["input"]) > 1:
            return httpx.Response(413, text="Payload Too Large")
        if body["input"][0].startswith("data:image/jpeg"):
            return httpx.Response(400, text="corrupt slide image")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.9, 0.1]}]})

    captured = _patch_client(monkeypatch, handler)
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    results = sorted(await provider.embed_images([good_png, bad_jpeg]), key=lambda r: r.index)

    # 1 batch attempt + 2 single-image fallback calls
    assert len(captured) == 3
    assert results[0].embedding == [0.9, 0.1] and results[0].error is None
    assert results[1].embedding is None and "corrupt slide image" in (results[1].error or "")


# ---------------------------------------------------------------------------
# Rate-limit deferral (with and without Retry-After) and fatal propagations
# ---------------------------------------------------------------------------


async def test_http_429_without_retry_after_slows_the_key_and_defers_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_b64 = _make_image_b64("PNG")

    def handler(_req: httpx.Request, _body: dict) -> httpx.Response:
        # NVIDIA Build often returns 429 with NO Retry-After header.
        return httpx.Response(429, json={"status": 429, "title": "Too Many Requests"})

    _patch_client(monkeypatch, handler)
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    pacer = MagicMock()
    pacer.acquire = AsyncMock()
    provider._pacer = pacer

    with pytest.raises(EmbeddingRateLimited) as raised:
        await provider.embed_images([png_b64])
    assert raised.value.retry_after == 60.0
    pacer.acquire.assert_awaited_with(Lane.BULK)
    pacer.penalize.assert_called_with(15.0)


async def test_a_saturated_pacer_defers_images_without_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_b64 = _make_image_b64("PNG")
    sent: list[httpx.Request] = []

    def handler(req: httpx.Request, _body: dict) -> httpx.Response:
        sent.append(req)
        return httpx.Response(200, json={"data": []})

    _patch_client(monkeypatch, handler)
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    pacer = MagicMock()
    pacer.acquire = AsyncMock(side_effect=PacerBusy(90.0))
    provider._pacer = pacer

    with pytest.raises(EmbeddingRateLimited) as raised:
        await provider.embed_images([png_b64])
    assert raised.value.retry_after == 90.0
    assert sent == []


async def test_non_retryable_and_send_allowance_spent_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_b64 = _make_image_b64("PNG")
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")

    class FakeGateway:
        def __init__(self, exc: BaseException) -> None:
            self._exc = exc

        async def call(self, _fn) -> object:
            raise self._exc

    monkeypatch.setattr(
        "app.services.embeddings.multimodal.nvidia_provider.get_embedding_gateway",
        lambda: FakeGateway(NonRetryableError("quota gone")),
    )
    with pytest.raises(NonRetryableError):
        await provider.embed_images([png_b64])

    monkeypatch.setattr(
        "app.services.embeddings.multimodal.nvidia_provider.get_embedding_gateway",
        lambda: FakeGateway(SendAllowanceSpent("budget spent")),
    )
    with pytest.raises(SendAllowanceSpent):
        await provider.embed_images([png_b64])


async def test_empty_and_all_invalid_batches_make_no_http_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_client(
        monkeypatch,
        lambda _req, _body: httpx.Response(500, text="should not be called"),
    )
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    assert await provider.embed_images([]) == []
    invalid_only = await provider.embed_images(["not-valid-b64!!!"])
    assert len(invalid_only) == 1
    assert invalid_only[0].embedding is None
    assert invalid_only[0].error == "invalid image data"
    assert captured == []


async def test_server_500_reports_per_image_errors_without_split_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_b64 = _make_image_b64("PNG")
    jpeg_b64 = _make_image_b64("JPEG")
    captured = _patch_client(
        monkeypatch,
        lambda _req, _body: httpx.Response(500, text="upstream outage"),
    )
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    results = await provider.embed_images([png_b64, jpeg_b64])
    assert len(captured) == 1
    assert [r.index for r in results] == [0, 1]
    assert all(r.embedding is None and "upstream outage" in (r.error or "") for r in results)


async def test_oversized_pixel_count_raster_is_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.embeddings.multimodal.nvidia_provider.MAX_DECODED_PIXELS",
        16,
    )
    oversized_bmp = _make_image_b64("BMP", size=(8, 8))  # 64 pixels > 16 limit
    captured = _patch_client(
        monkeypatch,
        lambda _req, _body: httpx.Response(500, text="should not be called"),
    )
    provider = NvidiaMultimodalProvider(api_key="nvapi-k", model_name="nvidia/nemotron-3.5-embed-8b")
    results = await provider.embed_images([oversized_bmp])
    assert captured == []
    assert len(results) == 1
    assert results[0].embedding is None
    assert results[0].error == "invalid image data"
