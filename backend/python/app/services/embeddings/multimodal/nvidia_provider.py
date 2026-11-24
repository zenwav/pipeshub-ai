"""NVIDIA NIM native multimodal image embedding (for example, nvidia/nemotron-3.5-embed-8b).

NVIDIA NIM's multimodal ``/v1/embeddings`` contract differs from generic
OpenAI-compatible servers in three ways:
- ``input_type="passage"`` and ``modality="image"`` are required for every image
  request, and ``truncate="END"`` prevents over-length rejections.
- Only PNG and JPEG data URIs are supported natively, and declared MIME types
  must match the actual image bytes; other raster formats (BMP, WebP, GIF,
  TIFF, ICO) are converted losslessly to RGB PNG.
- Hosted NVIDIA Build keys allow a fixed number of requests a minute and
  frequently omit ``Retry-After`` on HTTP 429, so requests share the indexing
  lane of the container-wide pacer (``app.utils.nvidia_pacer``) and defer the
  record on exhausted 429s instead of silently skipping slide images.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import TYPE_CHECKING, override

import httpx
from PIL import Image

from app.exceptions.non_retryable import NonRetryableError
from app.services.embeddings.multimodal._concurrent import gather_or_cancel
from app.services.embeddings.multimodal._response import (
    describe_request_error,
    map_embedding_response,
)
from app.services.embeddings.multimodal.interface import (
    ImageEmbeddingResult,
    IMultimodalEmbeddingProvider,
)
from app.services.llm_gateway import wire
from app.services.llm_gateway.embedding_gateway import (
    EmbeddingRateLimited,
    get_embedding_gateway,
)
from app.services.llm_gateway.wire import SendAllowanceSpent, refused_by_allowance
from app.utils.aimodels import parse_embedding_dimensions, require_allowed_endpoint
from app.utils.image_utils import MAX_DECODED_PIXELS, get_mime_type_from_base64
from app.utils.model_egress import guarded_async_client
from app.utils.nvidia_embeddings import (
    DEFAULT_NVIDIA_EMBEDDING_MODEL,
    DEFAULT_RATE_LIMIT_COOLDOWN_S,
    DEFAULT_RATE_LIMIT_DEFER_S,
    get_nvidia_endpoint_pacer,
    normalize_nvidia_base_url,
    pacer_busy_error,
    retry_after_from_exception,
)
from app.utils.nvidia_pacer import Lane, PacerBusy

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

_CONCURRENCY_LIMIT = 2
_MAX_BATCH_IMAGES = 4
_MAX_BATCH_PAYLOAD_BYTES = 4 * 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 60.0
_PASSTHROUGH_IMAGE_MIMES = frozenset({"image/png", "image/jpeg"})
_SPLIT_FALLBACK_STATUSES = frozenset({400, 413, 422})


def _group_image_batches(
    items: list[tuple[int, str]],
    *,
    max_images: int = _MAX_BATCH_IMAGES,
    max_bytes: int = _MAX_BATCH_PAYLOAD_BYTES,
) -> list[list[tuple[int, str]]]:
    """Group ``(index, data_uri)`` pairs into count- and byte-bounded batches."""
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_bytes = 0
    for item in items:
        item_bytes = len(item[1])
        if current and (
            len(current) >= max_images or current_bytes + item_bytes > max_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
    if current:
        batches.append(current)
    return batches


def _convert_to_rgb_png_uri(normalized_b64: str) -> str | None:
    """Decode non-PNG/JPEG raster bytes and re-encode losslessly as an RGB PNG data URI."""
    try:
        raw = base64.b64decode(normalized_b64)
        with Image.open(io.BytesIO(raw)) as source:
            if getattr(source, "is_animated", False):
                return None
            width, height = source.size
            if width * height > MAX_DECODED_PIXELS:
                return None
            if source.mode in ("RGBA", "LA", "PA") or "transparency" in source.info:
                rgba = source.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = source.convert("RGB") if source.mode != "RGB" else source.copy()
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
    except Exception:
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def _prepare_image_data_uri(normalized_b64: str) -> str | None:
    """Build a verified PNG or JPEG data URI from normalized base64 bytes."""
    try:
        mime = get_mime_type_from_base64(normalized_b64)
    except Exception:
        return None
    if mime in _PASSTHROUGH_IMAGE_MIMES:
        return f"data:{mime};base64,{normalized_b64}"
    return await asyncio.to_thread(_convert_to_rgb_png_uri, normalized_b64)


class NvidiaMultimodalProvider(IMultimodalEmbeddingProvider):
    """Native NVIDIA NIM multimodal image embedding provider."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str | None,
        base_url: str | None = None,
        dimensions: int | str | None = None,
        normalize_fn: Callable[[str], object] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model_name = (model_name or "").strip() or DEFAULT_NVIDIA_EMBEDDING_MODEL
        self.base_url = normalize_nvidia_base_url(base_url)
        require_allowed_endpoint(self.base_url)
        self.dimensions = parse_embedding_dimensions(dimensions)
        # Consumed by IMultimodalEmbeddingProvider.normalize().
        self._normalize_fn = normalize_fn
        self.logger = logger
        self._pacer = get_nvidia_endpoint_pacer(self.base_url, api_key)

    @property
    @override
    def provider_name(self) -> str:
        return "nvidia"

    def _build_payload(self, image_uris: list[str]) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model_name,
            "input": image_uris,
            "encoding_format": "float",
            "input_type": "passage",
            "modality": "image",
            "truncate": "END",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    async def _post_uris(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: dict[str, str],
        image_uris: list[str],
    ) -> object:
        payload = self._build_payload(image_uris)

        async def make_call() -> object:
            try:
                await self._pacer.acquire(Lane.BULK)
            except PacerBusy as busy:
                raise pacer_busy_error(busy) from busy
            try:
                resp = await client.post(endpoint, headers=headers, json=payload)
                _ = resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    cooldown = (
                        retry_after_from_exception(exc) or DEFAULT_RATE_LIMIT_COOLDOWN_S
                    )
                    self._pacer.penalize(cooldown)
                raise
            return resp.json().get("data", [])

        try:
            return await get_embedding_gateway().call(make_call)
        except EmbeddingRateLimited:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = (
                    retry_after_from_exception(exc)
                    or DEFAULT_RATE_LIMIT_DEFER_S
                )
                raise EmbeddingRateLimited(
                    "NVIDIA NIM rate limit reached; deferring record",
                    retry_after=wait,
                ) from exc
            raise

    async def _embed_batch_individually(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: dict[str, str],
        batch: list[tuple[int, str]],
    ) -> list[ImageEmbeddingResult]:
        results: list[ImageEmbeddingResult] = []
        for idx, uri in batch:
            try:
                data = await self._post_uris(client, endpoint, headers, [uri])
            except (EmbeddingRateLimited, NonRetryableError, SendAllowanceSpent):
                raise
            except Exception as err:
                if refused_by_allowance(err):
                    raise
                if self.logger:
                    self.logger.warning(
                        "NVIDIA NIM image %d failed: %s",
                        idx,
                        describe_request_error(err),
                    )
                results.append(
                    ImageEmbeddingResult(index=idx, error=describe_request_error(err))
                )
            else:
                results.extend(map_embedding_response(data, [idx]))
        return results

    @override
    async def embed_images(self, image_base64s: list[str]) -> list[ImageEmbeddingResult]:
        if not image_base64s:
            return []

        valid_items: list[tuple[int, str]] = []
        invalid_results: list[ImageEmbeddingResult] = []
        for idx, raw_image in enumerate(image_base64s):
            normalized = await self.normalize(raw_image)
            data_uri = (
                await _prepare_image_data_uri(normalized)
                if isinstance(normalized, str) and normalized
                else None
            )
            if data_uri is None:
                invalid_results.append(
                    ImageEmbeddingResult(index=idx, error="invalid image data")
                )
            else:
                valid_items.append((idx, data_uri))

        if not valid_items:
            return invalid_results

        endpoint = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        batches = _group_image_batches(valid_items)
        semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def process_batch(
            client: httpx.AsyncClient,
            batch: list[tuple[int, str]],
        ) -> list[ImageEmbeddingResult]:
            async with semaphore:
                request_indices = [idx for idx, _ in batch]
                uris = [uri for _, uri in batch]
                try:
                    data = await self._post_uris(client, endpoint, headers, uris)
                except (EmbeddingRateLimited, NonRetryableError, SendAllowanceSpent):
                    raise
                except httpx.HTTPStatusError as err:
                    if refused_by_allowance(err):
                        raise
                    if (
                        len(batch) > 1
                        and err.response.status_code in _SPLIT_FALLBACK_STATUSES
                    ):
                        return await self._embed_batch_individually(
                            client, endpoint, headers, batch
                        )
                    if self.logger:
                        self.logger.warning(
                            "NVIDIA NIM image batch failed: %s",
                            describe_request_error(err),
                        )
                    return [
                        ImageEmbeddingResult(index=idx, error=describe_request_error(err))
                        for idx, _ in batch
                    ]
                except Exception as err:
                    if refused_by_allowance(err):
                        raise
                    if self.logger:
                        self.logger.warning(
                            "NVIDIA NIM image batch failed: %s",
                            describe_request_error(err),
                        )
                    return [
                        ImageEmbeddingResult(index=idx, error=describe_request_error(err))
                        for idx, _ in batch
                    ]
                return map_embedding_response(data, request_indices)

        async with guarded_async_client(
            self.base_url, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as client:
            _ = wire.install_wire_hook(client)
            batch_results = await gather_or_cancel(
                [process_batch(client, batch) for batch in batches]
            )

        flattened: list[ImageEmbeddingResult] = list(invalid_results)
        for group in batch_results:
            flattened.extend(group)
        flattened.sort(key=lambda result: result.index)
        return flattened
