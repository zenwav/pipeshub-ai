"""Native NVIDIA NIM embedding client with asymmetric input_type and container-wide request pacing.

NVIDIA NIM asymmetric embedding models (such as ``nvidia/nemotron-3.5-embed-8b``)
require ``input_type="passage"`` when indexing documents and ``input_type="query"``
when embedding search queries, and support ``truncate="END"`` and Matryoshka
``dimensions``. Hosted NVIDIA Build keys allow a fixed number of requests a minute
and often answer 429 without ``Retry-After``, so text and image calls for one
endpoint and key are paced together across processes (``app.utils.nvidia_pacer``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast, override
from urllib.parse import urlparse

import openai
from langchain_core.embeddings import Embeddings
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)

from app.config.constants.ai_models import NVIDIA_NIM_BASE_URL
from app.services.llm_gateway.embedding_gateway import EmbeddingRateLimited
from app.services.llm_gateway.wire import parse_retry_after
from app.utils.embedding_retry import is_quota_exhausted
from app.utils.logger import create_logger
from app.utils.nvidia_pacer import (
    Lane,
    NvidiaPacer,
    PacerBusy,
    PacerLimits,
    get_nvidia_pacer,
)

if TYPE_CHECKING:
    from typing import Self

logger = create_logger("nvidia_embeddings")

DEFAULT_NVIDIA_EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-8b"
# NVIDIA Build's free tier allows 40 requests a minute per key; 36 leaves room for the
# difference between NVIDIA's window and ours.
DEFAULT_HOSTED_NVIDIA_RPM = 36
# Of those, the share indexing may not take, so a search is not queued behind documents.
DEFAULT_NVIDIA_INTERACTIVE_RESERVE_RPM = 6
DEFAULT_RATE_LIMIT_COOLDOWN_S = 15.0
# How long a record waits before it is sent again after a 429 that named no wait.
DEFAULT_RATE_LIMIT_DEFER_S = 60.0
_NVIDIA_DEFAULT_HOST = "integrate.api.nvidia.com"
_NVIDIA_HOST_SUFFIX = ".api.nvidia.com"
_RPM_ENV = "NVIDIA_NIM_RPM"
_RESERVE_ENV = "NVIDIA_NIM_INTERACTIVE_RESERVE_RPM"


def normalize_nvidia_base_url(endpoint: str | None) -> str:
    """Return a clean NVIDIA NIM base URL without trailing slashes."""
    raw = (endpoint or "").strip().rstrip("/")
    return raw or NVIDIA_NIM_BASE_URL


def is_nvidia_hosted_endpoint(endpoint: str | None) -> bool:
    """Whether ``endpoint`` targets NVIDIA's hosted Build API rather than a self-hosted NIM."""
    raw = (endpoint or "").strip()
    if not raw:
        return True
    try:
        host = (urlparse(raw if "://" in raw else f"//{raw}").hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host == _NVIDIA_DEFAULT_HOST or host.endswith(_NVIDIA_HOST_SUFFIX)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(float(raw)))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default


def nvidia_pacer_limits(base_url: str) -> PacerLimits:
    """The request limits for ``base_url``: NVIDIA Build's by default, none for a self-hosted NIM."""
    default_total = DEFAULT_HOSTED_NVIDIA_RPM if is_nvidia_hosted_endpoint(base_url) else 0
    return PacerLimits.of(
        _env_int(_RPM_ENV, default_total),
        _env_int(_RESERVE_ENV, DEFAULT_NVIDIA_INTERACTIVE_RESERVE_RPM),
    )


def get_nvidia_endpoint_pacer(base_url: str | None, api_key: str | None) -> NvidiaPacer:
    """The pacer text and image calls for this endpoint and key share, in every process of the container."""
    normalized = normalize_nvidia_base_url(base_url)
    return get_nvidia_pacer(normalized, api_key, nvidia_pacer_limits(normalized))


def retry_after_from_exception(exc: BaseException) -> float | None:
    """Extract Retry-After seconds from an OpenAI or httpx exception if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return parse_retry_after(headers) if headers is not None else None


def is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def pacer_busy_error(busy: PacerBusy) -> EmbeddingRateLimited:
    """Hand the work back for the pacer's wait instead of sleeping past the caller's deadline."""
    return EmbeddingRateLimited(
        "NVIDIA NIM request limit is in use; deferring", retry_after=max(busy.wait_s, 1.0),
    )


def _extract_vectors(response: object, expected_count: int) -> list[list[float]]:
    """Parse, order by index, and validate an OpenAI-style embeddings response."""
    raw_data: object = (
        cast("dict[str, object]", response).get("data")
        if isinstance(response, dict)
        else getattr(response, "data", None)
    )
    if not isinstance(raw_data, list):
        raise ValueError("NVIDIA NIM returned a malformed embeddings response without a data list")
    items = cast("list[object]", raw_data)
    indexed: list[tuple[int, list[float]]] = []
    for arrival, item in enumerate(items):
        if isinstance(item, dict):
            item_dict = cast("dict[str, object]", item)
            idx = item_dict.get("index")
            embedding = item_dict.get("embedding")
        else:
            idx = getattr(item, "index", None)
            embedding = getattr(item, "embedding", None)
        position = idx if isinstance(idx, int) and not isinstance(idx, bool) else arrival
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"NVIDIA NIM returned an empty embedding at index {position}")
        raw_vec = cast("list[object]", embedding)
        vec: list[float] = []
        for coord in raw_vec:
            if isinstance(coord, bool) or not isinstance(coord, (int, float)):
                raise ValueError(
                    f"NVIDIA NIM returned a non-numeric embedding coordinate at index {position}"
                )
            vec.append(float(coord))
        indexed.append((position, vec))
    if len(indexed) != expected_count:
        raise ValueError(
            f"NVIDIA NIM returned an unexpected embedding count: "
            f"expected {expected_count} embeddings, got {len(indexed)}"
        )
    # A repeated or out-of-range index would still pass the count and attach vectors to the wrong texts.
    if sorted(position for position, _ in indexed) != list(range(expected_count)):
        raise ValueError("NVIDIA NIM returned embedding indexes that do not match the request")
    indexed.sort(key=lambda pair: pair[0])
    return [vec for _, vec in indexed]


class NvidiaEmbeddings(BaseModel, Embeddings):
    """NVIDIA NIM embeddings backed by the OpenAI SDK with asymmetric ``input_type``."""

    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)
    model: str = DEFAULT_NVIDIA_EMBEDDING_MODEL
    dimensions: int | None = None
    truncate: Literal["NONE", "END"] = "END"
    batch_size: int = 50
    nvidia_api_key: SecretStr | None = Field(alias="api_key", default=None)
    nvidia_api_base: str = Field(default=NVIDIA_NIM_BASE_URL, alias="base_url")
    max_retries: int = 0
    request_timeout: float | tuple[float, float] | None = Field(default=None, alias="timeout")
    default_headers: dict[str, str] | None = None
    default_query: dict[str, object] | None = None
    http_client: Any = None
    http_async_client: Any = None

    _pacer: NvidiaPacer | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.nvidia_api_base = normalize_nvidia_base_url(self.nvidia_api_base)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.dimensions is not None and self.dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {self.dimensions}")

        api_key = self.nvidia_api_key.get_secret_value() if self.nvidia_api_key else None
        client_params: dict[str, object] = {
            "api_key": api_key,
            "base_url": self.nvidia_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not self.client:
            sync_extra: dict[str, object] = (
                {"http_client": self.http_client} if self.http_client is not None else {}
            )
            self.client = openai.OpenAI(**client_params, **sync_extra).embeddings  # type: ignore[arg-type]
        if not self.async_client:
            async_extra: dict[str, object] = (
                {"http_client": self.http_async_client} if self.http_async_client is not None else {}
            )
            self.async_client = openai.AsyncOpenAI(**client_params, **async_extra).embeddings  # type: ignore[arg-type]
        self._pacer = get_nvidia_endpoint_pacer(self.nvidia_api_base, api_key)
        return self

    @property
    def base_url(self) -> str:
        """Expose ``base_url`` for ``llm_gateway.provider_key`` and dimension-probe caching."""
        return self.nvidia_api_base

    def _pacer_for_endpoint(self) -> NvidiaPacer:
        if self._pacer is None:
            api_key = self.nvidia_api_key.get_secret_value() if self.nvidia_api_key else None
            self._pacer = get_nvidia_endpoint_pacer(self.nvidia_api_base, api_key)
        return self._pacer

    def _on_failure(self, exc: Exception, input_type: Literal["passage", "query"]) -> None:
        """Slow every caller of this key after a 429; a document batch is handed back instead of failing."""
        if not is_rate_limit_error(exc) or is_quota_exhausted(exc):
            return
        explicit_wait = retry_after_from_exception(exc)
        self._pacer_for_endpoint().penalize(explicit_wait or DEFAULT_RATE_LIMIT_COOLDOWN_S)
        # With a Retry-After the caller's own policy decides (retry, or hand the record back). Without one,
        # chaining EmbeddingRateLimited from exc lets VectorStore run its normal retries across the cooldown
        # above, and defers the record once retries run out instead of marking it failed. A search raises exc
        # directly: it is a person waiting, not background indexing work to defer.
        if explicit_wait is None and input_type == "passage":
            raise EmbeddingRateLimited(
                "NVIDIA NIM answered 429 without Retry-After; deferring",
                retry_after=DEFAULT_RATE_LIMIT_DEFER_S,
            ) from exc

    def _busy(self, busy: PacerBusy) -> NoReturn:
        raise pacer_busy_error(busy) from busy

    def _request_kwargs(
        self, texts: list[str], *, input_type: Literal["passage", "query"]
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
            "extra_body": {
                "input_type": input_type,
                "truncate": self.truncate,
            },
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    @staticmethod
    def _lane(input_type: Literal["passage", "query"]) -> Lane:
        return Lane.BULK if input_type == "passage" else Lane.INTERACTIVE

    def _embed_sync(
        self, texts: list[str], *, input_type: Literal["passage", "query"]
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        pacer = self._pacer_for_endpoint()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                pacer.acquire_sync(self._lane(input_type))
            except PacerBusy as busy:
                self._busy(busy)
            try:
                response = self.client.create(**self._request_kwargs(batch, input_type=input_type))
            except Exception as exc:
                self._on_failure(exc, input_type)
                raise
            vectors.extend(_extract_vectors(response, len(batch)))
        return vectors

    async def _embed_async(
        self, texts: list[str], *, input_type: Literal["passage", "query"]
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        pacer = self._pacer_for_endpoint()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                await pacer.acquire(self._lane(input_type))
            except PacerBusy as busy:
                self._busy(busy)
            try:
                response = await self.async_client.create(
                    **self._request_kwargs(batch, input_type=input_type)
                )
            except Exception as exc:
                self._on_failure(exc, input_type)
                raise
            vectors.extend(_extract_vectors(response, len(batch)))
        return vectors

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_sync(texts, input_type="passage")

    @override
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_async(texts, input_type="passage")

    @override
    def embed_query(self, text: str) -> list[float]:
        return self._embed_sync([text], input_type="query")[0]

    @override
    async def aembed_query(self, text: str) -> list[float]:
        return (await self._embed_async([text], input_type="query"))[0]
