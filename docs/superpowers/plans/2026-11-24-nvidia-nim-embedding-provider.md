# Native NVIDIA NIM multimodal embedding provider implementation plan

> **Revised 2026-11-25 (review fixes).** Where this document and the code differ, the code is authoritative:
> - Pacing moved to `app/utils/nvidia_pacer.py`: one state file per endpoint and key, shared by every process in the container under `flock` (the query and indexing services are separate processes, so a per-process limiter let each spend the whole allowance). Defaults: 36 requests a minute in total on NVIDIA Build, of which indexing may use 30, leaving 6 for searches (`NVIDIA_NIM_RPM`, `NVIDIA_NIM_INTERACTIVE_RESERVE_RPM`). Nothing is booked ahead, so a cancelled caller leaves nothing behind; a wait longer than 45 s hands the work back as `EmbeddingRateLimited` instead of sleeping past the batch deadline.
> - A 429 without `Retry-After` on document indexing raises `EmbeddingRateLimited` (record deferred 60 s) instead of writing a fake `Retry-After` header into NVIDIA's response. Searches keep the caller's retries.
> - Response vectors are refused when their indexes are repeated or out of range.
> - The provider reuses the shared embedding fields (`EMBEDDING_COMMON_TAIL`): Multimodal defaults to off, as for every other embedding provider. The icon is `nvidia-color.svg`.
> - Decision-model rejections keep the LLM hint (reasoning effort, context length).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline execution, per user preference to skip subagent review and let the user review with Carbon and Claude Code). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `nvidia` (`NVIDIA NIM`) multimodal asymmetric embedding provider for `nvidia/nemotron-3.5-embed-8b`, fix embedding health-check error messages from issue #6333, add a manual GitHub Actions ARM64 Docker Hub build workflow, and document the zero-data-loss OCI VPS upgrade procedure.

**Architecture:** Build `NvidiaEmbeddings(BaseModel, Embeddings)` directly on `openai.OpenAI` / `AsyncOpenAI` (reusing `openai==2.45.0` and PipesHub's `_OPENAI_RESOURCE_CLIENTS` egress, identity, and wire hooks) sending `input_type="passage"` for documents and `"query"` for search queries with `truncate="END"`. Pair it with `NvidiaMultimodalProvider(IMultimodalEmbeddingProvider)` sending 1 normalized PNG/JPEG image per request with `input_type="passage"`, `modality="image"`, optional matching `dimensions`, and `EmbeddingGateway` rate-limit deferral. Keep the fork CI workflow in a standalone file (`.github/workflows/fork-docker-arm64.yml`) so upstream rebases stay clean.

**Tech Stack:** Python 3.12, Pydantic v2, `openai==2.45.0`, `httpx==0.28.1`, `Pillow`, `langchain-core==1.7.0`, pytest, Ruff, TypeScript/Next.js frontend, GitHub Actions (`ubuntu-24.04-arm`, Docker Buildx).

**Spec:** [2026-11-24-nvidia-nim-embedding-provider-design.md](../specs/2026-11-24-nvidia-nim-embedding-provider-design.md)

## Global constraints

- Embedding-only capability for `nvidia` (`ModelCapability.EMBEDDING`); no LLM, TTS, STT, image generation, or decision capability.
- Zero new Python dependencies in `backend/python/pyproject.toml`; reuse `openai`, `httpx`, `Pillow`, and `langchain-core` already in `pipeshubai/pipeshub-ai-base:python-deps-slim`.
- Strict Ruff compliance (`backend/python/pyproject.toml`, including `ANN001`-`ANN206`, `ANN401`, `UP006`/`UP007`/`UP035`, `TC001`-`TC003`, `FBT001`/`FBT002`, `B904`).
- Preserve PipesHub's self-hosted error-isolation contract (`openAICompatible`, `litellmProxy`, `ollama`, and custom non-NVIDIA NIM endpoints must never leak raw upstream response bodies into the setup dialog).
- Do not edit upstream `.github/workflows/release-publish.yml`.
- Stage changes locally (`git add`), do not run `git commit` (blocked by user's interactive GPG pinentry; user will trigger signing after Carbon/Claude review), and never run `git push` without explicit user approval.
- Skip automated subagent review; validate with unit tests, coverage checks, and linters locally, then provide a Claude Code review prompt.

## Review focus

1. **Blank or trailing-slash `endpoint` URLs:** Blank/whitespace `endpoint` must fall back to `https://integrate.api.nvidia.com/v1`, and trailing slashes must be stripped so multimodal POSTs hit `{base}/embeddings` rather than `{base}//embeddings`. (Pinned in Tasks 1 and 3.)
2. **Mislabeled image data-URI prefixes and non-PNG/JPEG formats:** A PNG mislabeled as `data:image/jpeg;base64,...` or a WebP/GIF/BMP/RGBA image from lecture slides must be identified by actual magic bytes (`get_mime_type_from_base64`) and, when neither PNG nor JPEG, converted to lossless RGB PNG via Pillow without crashing sibling images on corrupt input. (Pinned in Task 3.)
3. **Out-of-order or truncated embedding arrays:** If NIM returns `data` items out of index order or returns fewer vectors than requested (or an empty vector), `NvidiaEmbeddings` must sort by `index` and raise `ValueError` rather than silently misaligning slide chunks in Qdrant. (Pinned in Task 1.)
4. **Malformed or blank `dimensions` configuration values:** Empty string `""`, `0`, negative numbers, or non-numeric strings must be ignored (omitting `"dimensions"` so native 4096 applies), while valid string integers like `"2048"` must parse cleanly to `2048` on both text and image paths. (Pinned in Tasks 1 and 3.)
5. **Mid-deck HTTP 429 rate limits (with or without `Retry-After`) vs single-image errors:** Free NVIDIA Build keys cap at 40 RPM and often omit `Retry-After` on 429s. A shared smooth RPM pacer (`NvidiaRateLimiter`, default 35 RPM on hosted `*.api.nvidia.com`, 0 on self-hosted NIM, configurable via `NVIDIA_NIM_RPM`) must pace both text and image calls evenly, apply a cooldown penalty on 429, and convert exhausted 429s (even without `Retry-After`) into `EmbeddingRateLimited(retry_after=60.0)` so PipesHub automatically defers and retries the record without manual Reindex clicks. Meanwhile, a single corrupt image or 400/413/422 in a multi-image batch must fall back 1-by-1 and isolate only the failing slide. (Pinned in Tasks 1 and 3.)

---

## File structure

- `backend/python/app/config/constants/ai_models.py`: add `NVIDIA_NIM_BASE_URL`.
- `backend/python/app/utils/nvidia_embeddings.py`: new focused module holding `NvidiaEmbeddings` and response alignment helpers.
- `backend/python/app/utils/aimodels.py`: add `EmbeddingProvider.NVIDIA`, `SELF_HOSTABLE_EMBEDDING_PROVIDERS` entry, and `_get_embedding_model_impl` branch.
- `backend/python/app/config/ai_models/providers/nvidia.py`: new `NvidiaProvider` registry declaration.
- `backend/python/app/config/ai_models/providers/__init__.py`: import and register `NvidiaProvider`.
- `backend/python/app/services/ai_models/discovery/strategies.py`: register static `"nvidia"` strategy for `nvidia/nemotron-3.5-embed-8b`.
- `backend/python/app/services/resource_governor/explain.py`: add `"nvidia"` to `EMBEDDING_PROVIDER_NAMES`.
- `backend/python/app/services/embeddings/multimodal/config.py`: add optional `dimensions: int | None = None` to `MultimodalProviderConfig`.
- `backend/python/app/services/embeddings/multimodal/nvidia_provider.py`: new `NvidiaMultimodalProvider`.
- `backend/python/app/services/embeddings/multimodal/factory.py`: register `EmbeddingProvider.NVIDIA.value`.
- `backend/python/app/modules/transformers/vectorstore.py`: parse positive `self.configured_dimensions` and forward to `MultimodalProviderConfig`.
- `backend/python/app/api/routes/health.py`: forward parsed `dimensions` to `_probe_image_embedding`, fix `_rejected_settings_message` per `kind`, add `"nvidia"` to `_HOSTED_PROVIDER_KEYS`, `_answers_from_vendor`, and `_llm_health_check_needs_outbound`.
- `frontend/public/icons/ai-models/nvidia.svg`: NVIDIA brand SVG icon.
- `frontend/lib/utils/llm-provider-icons.ts`: map `nvidia: 'nvidia.svg'`.
- `frontend/app/(main)/chat/constants.ts`: map `nvidia: 'NVIDIA NIM'`.
- `frontend/app/(main)/onboarding/types.ts`: add `'nvidia'` to `EmbeddingProviderType`.
- `.github/workflows/fork-docker-arm64.yml`: manual `workflow_dispatch` native `linux/arm64` slim Docker Hub build workflow.
- `docs/superpowers/runbooks/oci-arm64-fork-upgrade.md`: zero-data-loss OCI Ubuntu 26.04 upgrade and rollback runbook.

---

### Task 1: Text embedding client (`NvidiaEmbeddings`) and `aimodels.py` factory wiring

**Files:**
- Create: `backend/python/app/utils/nvidia_embeddings.py`
- Modify: `backend/python/app/config/constants/ai_models.py:33-42`
- Modify: `backend/python/app/utils/aimodels.py:124-165, 965-975`
- Test: `backend/python/tests/unit/utils/test_nvidia_embeddings.py`

**Interfaces:**
- Consumes: `app.utils.aimodels._guard_clients`, `_set_embedding_dimensions_kwarg`, `_set_openai_client_limits_kwargs`, `require_allowed_endpoint`, `_OPENAI_RESOURCE_CLIENTS`.
- Produces:
  - `app.config.constants.ai_models.NVIDIA_NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"`
  - `app.utils.aimodels.EmbeddingProvider.NVIDIA = "nvidia"`
  - `app.utils.nvidia_embeddings.NvidiaRateLimiter` + `get_nvidia_rate_limiter(base_url: str, api_key: str | None) -> NvidiaRateLimiter` (smooth interval pacer, default 35 RPM on hosted `*.api.nvidia.com`, 0 on self-hosted NIM, overridable via `NVIDIA_NIM_RPM`, with `acquire()`, `acquire_sync()`, and `penalize(cooldown_s: float)`).
  - `app.utils.nvidia_embeddings.NvidiaEmbeddings(BaseModel, Embeddings)` with `embed_documents(texts: list[str]) -> list[list[float]]`, `aembed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`, `aembed_query(text: str) -> list[float]`.
  - `app.utils.aimodels.parse_embedding_dimensions(raw: object) -> int | None` (shared positive-int parser so `aimodels.py`, `vectorstore.py`, and `health.py` parse `dimensions` identically).

- [ ] **Step 1: Prepare Python 3.12 test venv (one-time workspace setup)**

If `backend/python/.venv` does not exist, create it with `uv venv --python 3.12 backend/python/.venv` and install the unit-test dependencies (`pytest`, `pytest-asyncio`, `pytest-timeout`, `pytest-cov`, `ruff`, `openai==2.45.0`, `httpx==0.28.1`, `langchain-core==1.7.0`, `langchain-openai==1.7.2`, `pydantic`, `pillow`, `fastapi`, `jsonschema`, `tiktoken`, `tenacity`, `cachetools`, `requests`, `botocore`, `boto3`).

- [ ] **Step 2: Write failing tests in `backend/python/tests/unit/utils/test_nvidia_embeddings.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.config.constants.ai_models import NVIDIA_NIM_BASE_URL
from app.utils.aimodels import EmbeddingProvider, get_embedding_model, parse_embedding_dimensions
from app.utils.nvidia_embeddings import NvidiaEmbeddings


def _resp(vectors: list[tuple[int, list[float]]]) -> MagicMock:
    items = []
    for idx, vec in vectors:
        item = MagicMock()
        item.index = idx
        item.embedding = vec
        items.append(item)
    r = MagicMock()
    r.data = items
    return r


def test_parse_embedding_dimensions_handles_blank_zero_negative_and_strings():
    assert parse_embedding_dimensions(None) is None
    assert parse_embedding_dimensions("") is None
    assert parse_embedding_dimensions("   ") is None
    assert parse_embedding_dimensions(0) is None
    assert parse_embedding_dimensions(-5) is None
    assert parse_embedding_dimensions(True) is None
    assert parse_embedding_dimensions("abc") is None
    assert parse_embedding_dimensions("2048") == 2048
    assert parse_embedding_dimensions(1024) == 1024


def test_embed_documents_sends_passage_and_batches_in_order():
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test", batch_size=2)
    emb.client = MagicMock()
    # Return first batch out of order to verify index sorting
    emb.client.create.side_effect = [
        _resp([(1, [2.0, 2.0]), (0, [1.0, 1.0])]),
        _resp([(0, [3.0, 3.0])]),
    ]
    out = emb.embed_documents(["a", "b", "c"])
    assert out == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert emb.client.create.call_count == 2
    first_kwargs = emb.client.create.call_args_list[0].kwargs
    assert first_kwargs == {
        "model": "nvidia/nemotron-3.5-embed-8b",
        "input": ["a", "b"],
        "encoding_format": "float",
        "extra_body": {"input_type": "passage", "truncate": "END"},
    }


def test_embed_documents_empty_list_makes_no_call():
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    assert emb.embed_documents([]) == []
    emb.client.create.assert_not_called()


async def test_aembed_documents_empty_and_batched_with_dimensions():
    emb = NvidiaEmbeddings(
        model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test", dimensions=1024, batch_size=2
    )
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(return_value=_resp([(0, [0.5] * 1024)]))
    assert await emb.aembed_documents([]) == []
    out = await emb.aembed_documents(["slide 1"])
    assert len(out) == 1 and len(out[0]) == 1024
    kwargs = emb.async_client.create.call_args.kwargs
    assert kwargs["dimensions"] == 1024
    assert kwargs["extra_body"] == {"input_type": "passage", "truncate": "END"}


def test_embed_query_and_aembed_query_send_query_input_type():
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    emb.client.create.return_value = _resp([(0, [0.1, 0.2])])
    assert emb.embed_query("what is backpropagation?") == [0.1, 0.2]
    assert emb.client.create.call_args.kwargs["extra_body"] == {
        "input_type": "query",
        "truncate": "END",
    }
    assert emb.client.create.call_args.kwargs["input"] == ["what is backpropagation?"]


async def test_aembed_query_sends_query_input_type():
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test", dimensions=512)
    emb.async_client = MagicMock()
    emb.async_client.create = AsyncMock(return_value=_resp([(0, [0.3] * 512)]))
    vec = await emb.aembed_query("gradient descent")
    assert len(vec) == 512
    assert emb.async_client.create.call_args.kwargs["extra_body"] == {
        "input_type": "query",
        "truncate": "END",
    }
    assert emb.async_client.create.call_args.kwargs["dimensions"] == 512


def test_truncated_or_empty_response_raises_value_error():
    emb = NvidiaEmbeddings(model="nvidia/nemotron-3.5-embed-8b", api_key="nvapi-test")
    emb.client = MagicMock()
    emb.client.create.return_value = _resp([(0, [1.0])])
    with pytest.raises(ValueError, match="expected 2 embeddings, got 1"):
        emb.embed_documents(["a", "b"])
    emb.client.create.return_value = _resp([(0, [])])
    with pytest.raises(ValueError, match="empty embedding"):
        emb.embed_documents(["a"])


def test_factory_builds_nvidia_default_and_custom_endpoint():
    default_emb = get_embedding_model(
        EmbeddingProvider.NVIDIA.value,
        {"configuration": {"apiKey": "nvapi-k", "model": "nvidia/nemotron-3.5-embed-8b", "endpoint": "   "}, "isDefault": True},
    )
    assert isinstance(default_emb, NvidiaEmbeddings)
    assert default_emb.nvidia_api_base == NVIDIA_NIM_BASE_URL
    assert default_emb.max_retries == 0

    custom_emb = get_embedding_model(
        EmbeddingProvider.NVIDIA.value,
        {"configuration": {"apiKey": "nvapi-k", "model": "nvidia/nemotron-3.5-embed-8b", "endpoint": "https://nim.example.com/v1/", "dimensions": "2048"}, "isDefault": True},
    )
    assert isinstance(custom_emb, NvidiaEmbeddings)
    assert custom_emb.nvidia_api_base == "https://nim.example.com/v1"
    assert custom_emb.dimensions == 2048


def test_factory_blocks_metadata_endpoint():
    with pytest.raises(ValueError, match="link-local or cloud metadata"):
        get_embedding_model(
            EmbeddingProvider.NVIDIA.value,
            {"configuration": {"apiKey": "nvapi-k", "model": "nvidia/nemotron-3.5-embed-8b", "endpoint": "http://169.254.169.254/v1"}, "isDefault": True},
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `backend/python/.venv/Scripts/pytest.exe tests/unit/utils/test_nvidia_embeddings.py -v` (from `backend/python`).
Expected: FAIL (`ImportError` for `NVIDIA_NIM_BASE_URL` / `nvidia_embeddings`).

- [ ] **Step 4: Implement `NVIDIA_NIM_BASE_URL`, `NvidiaEmbeddings`, `parse_embedding_dimensions`, and `aimodels.py` wiring**

1. Add `NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"` in `backend/python/app/config/constants/ai_models.py` next to `OPENROUTER_BASE_URL`.
2. Create `backend/python/app/utils/nvidia_embeddings.py`:
   - `NvidiaEmbeddings(BaseModel, Embeddings)` with `model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())`.
   - Normalize `nvidia_api_base` by stripping whitespace and trailing `/` (falling back to `NVIDIA_NIM_BASE_URL` if blank).
   - Validate `batch_size >= 1` and `dimensions > 0` if set.
   - Build `openai.OpenAI(**client_params).embeddings` on `self.client` and `openai.AsyncOpenAI(**client_params).embeddings` on `self.async_client` in `@model_validator(mode="after")` (matching `OPENAI_RESOURCE_CLIENTS`).
   - Helper `_extract_vectors(response: object, expected_count: int) -> list[list[float]]` that supports SDK response objects and dicts, sorts by `index`, verifies exact `expected_count` and non-empty vector lists (raising `ValueError` otherwise).
   - Implement `embed_documents`, `aembed_documents` (`input_type="passage"`), `embed_query`, `aembed_query` (`input_type="query"`).
3. In `backend/python/app/utils/aimodels.py`:
   - Add `NVIDIA = "nvidia"` to `EmbeddingProvider` and `EmbeddingProvider.NVIDIA.value` to `SELF_HOSTABLE_EMBEDDING_PROVIDERS`.
   - Add `parse_embedding_dimensions(raw: object) -> int | None` (rejects `bool`, `None`, empty strings, `<= 0`, non-int strings; returns positive `int`) and use it in `_get_embedding_model_impl`.
   - Add `elif provider == EmbeddingProvider.NVIDIA.value:` in `_get_embedding_model_impl` resolving `endpoint = (configuration.get("endpoint") or "").strip().rstrip("/") or NVIDIA_NIM_BASE_URL`, applying `require_allowed_endpoint(endpoint)`, `_set_embedding_dimensions_kwarg`, `_set_openai_client_limits_kwargs`, and returning `_guard_clients(NvidiaEmbeddings(**nvidia_kwargs), _OPENAI_RESOURCE_CLIENTS, endpoint)`.

- [ ] **Step 5: Run tests and Ruff to verify Task 1 passes**

Run:
- `pytest tests/unit/utils/test_nvidia_embeddings.py tests/unit/utils/test_aimodels.py -v`
- `ruff check app/config/constants/ai_models.py app/utils/nvidia_embeddings.py app/utils/aimodels.py tests/unit/utils/test_nvidia_embeddings.py`
Expected: PASS, 0 Ruff errors.

- [ ] **Step 6: Stage Task 1 files**

Run: `git add backend/python/app/config/constants/ai_models.py backend/python/app/utils/nvidia_embeddings.py backend/python/app/utils/aimodels.py backend/python/tests/unit/utils/test_nvidia_embeddings.py`

---

### Task 2: Provider registry, static discovery, and cross-cutting Python invariants

**Files:**
- Create: `backend/python/app/config/ai_models/providers/nvidia.py`
- Modify: `backend/python/app/config/ai_models/providers/__init__.py`
- Modify: `backend/python/app/services/ai_models/discovery/strategies.py:1070-1112`
- Modify: `backend/python/app/services/resource_governor/explain.py:117-122`
- Modify: `backend/python/tests/unit/services/model_identity/test_provider_coverage.py:58-76`

**Interfaces:**
- Consumes: `EmbeddingProvider.NVIDIA`, `NVIDIA_NIM_BASE_URL`, `get_embedding_model` from Task 1.
- Produces: Registered `"nvidia"` provider in `ai_model_registry` and `discovery_registry`, normalized `"nvidia"` dependency in `explain.EMBEDDING_PROVIDER_NAMES`.

- [ ] **Step 1: Verify invariant tests currently fail for `EmbeddingProvider.NVIDIA`**

Run: `pytest tests/unit/services/load/test_projection.py::test_every_embedding_provider_maps_to_embedding tests/unit/services/model_identity/test_provider_coverage.py -v`
Expected: FAIL because `"nvidia"` is not yet in `EMBEDDING_PROVIDER_NAMES` and `test_provider_coverage.py`.

- [ ] **Step 2: Create `NvidiaProvider`, wire registry, discovery, `explain.py`, and `test_provider_coverage.py`**

1. Create `backend/python/app/config/ai_models/providers/nvidia.py`:
   - `@AIModelProviderBuilder("NVIDIA NIM", "nvidia")` with description `"NVIDIA NIM multimodal and asymmetric embedding models"`, capabilities `[ModelCapability.EMBEDDING]`, icon `"/icons/ai-models/nvidia.svg"`, color `"#76B900"`.
   - Fields: `API_KEY`, `_NVIDIA_MODEL` (`name="model"`, `default_value="nvidia/nemotron-3.5-embed-8b"`, `required=True`), `_NVIDIA_ENDPOINT` (`name="endpoint"`, `field_type="URL"`, `required=False`, placeholder `NVIDIA_NIM_BASE_URL`), `FRIENDLY_NAME`, `DIMENSIONS`, `_NVIDIA_MULTIMODAL` (`name="isMultimodal"`, `field_type="BOOLEAN"`, `required=False`, `default_value=True`).
2. Import and append `NvidiaProvider` to `ALL_PROVIDER_CLASSES` in `backend/python/app/config/ai_models/providers/__init__.py`.
3. In `backend/python/app/services/ai_models/discovery/strategies.py`, define `_NVIDIA = [RawModel(id="nvidia/nemotron-3.5-embed-8b", capabilities=["embedding"], is_multimodal=True)]` and add `"nvidia": _StaticStrategy(_NVIDIA)` in `build_registry()`.
4. Add `"nvidia"` to `EMBEDDING_PROVIDER_NAMES` in `backend/python/app/services/resource_governor/explain.py`.
5. Add `"nvidia"` to `test_embedding_clients_carry_the_hook` parametrize list and `known` set in `backend/python/tests/unit/services/model_identity/test_provider_coverage.py`.

- [ ] **Step 3: Add targeted assertions in `test_nvidia_embeddings.py` for registry and discovery**

Add tests asserting:
- `ai_model_registry.get_provider("nvidia")` has `capabilities == ["embedding"]`, `iconPath == "/icons/ai-models/nvidia.svg"`, `isMultimodal` default `True`, `model` default `"nvidia/nemotron-3.5-embed-8b"`, and `endpoint` optional.
- `discovery_registry.get("nvidia")` returns `nvidia/nemotron-3.5-embed-8b` with `capabilities == ["embedding"]` and `is_multimodal is True`.

- [ ] **Step 4: Run all contract and invariant test suites + Ruff**

Run:
- `pytest tests/unit/config/ai_models/test_ai_model_providers.py tests/unit/utils/test_ai_model_factory_contract.py tests/unit/services/ai_models/test_discovery.py tests/unit/services/load/test_projection.py tests/unit/services/model_identity/test_provider_coverage.py tests/unit/utils/test_nvidia_embeddings.py -v`
- `ruff check app/config/ai_models/providers/nvidia.py app/config/ai_models/providers/__init__.py app/services/ai_models/discovery/strategies.py app/services/resource_governor/explain.py tests/unit/services/model_identity/test_provider_coverage.py`
Expected: PASS, 0 Ruff errors.

- [ ] **Step 5: Stage Task 2 files**

Run: `git add backend/python/app/config/ai_models/providers/nvidia.py backend/python/app/config/ai_models/providers/__init__.py backend/python/app/services/ai_models/discovery/strategies.py backend/python/app/services/resource_governor/explain.py backend/python/tests/unit/services/model_identity/test_provider_coverage.py backend/python/tests/unit/utils/test_nvidia_embeddings.py`

---

### Task 3: Multimodal image embedding (`NvidiaMultimodalProvider`) and `dimensions` propagation

**Files:**
- Create: `backend/python/app/services/embeddings/multimodal/nvidia_provider.py`
- Modify: `backend/python/app/services/embeddings/multimodal/config.py:13-40`
- Modify: `backend/python/app/services/embeddings/multimodal/factory.py:11-95`
- Modify: `backend/python/app/modules/transformers/vectorstore.py:2465-2490, 2652-2674`
- Modify: `backend/python/app/api/routes/health.py:1820-1855`
- Test: `backend/python/tests/unit/services/embeddings/multimodal/test_nvidia_provider.py`
- Modify: `backend/python/tests/unit/services/embeddings/multimodal/test_factory.py`
- Modify: `backend/python/tests/unit/services/embeddings/test_image_providers_rate_limit.py`

**Interfaces:**
- Consumes: `NVIDIA_NIM_BASE_URL`, `EmbeddingProvider.NVIDIA`, `parse_embedding_dimensions` from Task 1; `IMultimodalEmbeddingProvider`, `ImageEmbeddingResult`, `map_embedding_response`, `describe_request_error`, `gather_or_cancel`, `get_embedding_gateway`, `EmbeddingRateLimited`, `NonRetryableError`, `wire.install_wire_hook`, `wire.refused_by_allowance`, `guarded_async_client`, `get_mime_type_from_base64`.
- Produces:
  - `MultimodalProviderConfig.dimensions: int | None = None`
  - `NvidiaMultimodalProvider(IMultimodalEmbeddingProvider)` registered in `MultimodalEmbeddingFactory` under `"nvidia"`.

- [ ] **Step 1: Write failing unit tests in `test_nvidia_provider.py`, `test_factory.py`, and `test_image_providers_rate_limit.py`**

In `test_nvidia_provider.py`, cover:
- `test_png_and_jpeg_preserve_magic_byte_mime_even_when_prefix_mislabeled`: passes a real 2x2 PNG bytes labeled `data:image/jpeg;base64,...` and a real 2x2 JPEG bytes; asserts requests hit `{base}/embeddings` (no double slash even when `base_url` had trailing `/`) with `data:image/png;base64,...` and `data:image/jpeg;base64,...`, `input_type="passage"`, `modality="image"`, `truncate="END"`, `encoding_format="float"`, and 1 image per HTTP call.
- `test_webp_and_bmp_convert_to_rgb_png_losslessly`: passes real 2x2 BMP and WebP base64 images; asserts sent `input[0]` starts with `data:image/png;base64,` and decodes as valid RGB PNG.
- `test_dimensions_forwarded_only_when_positive_int`: asserts `"dimensions"` omitted when `dimensions=None` and included as `"dimensions": 1024` when `dimensions=1024`.
- `test_corrupt_image_isolated_from_valid_siblings`: `[valid_png, "not-valid-b64!!!", valid_jpeg]` returns embeddings for index 0 and 2 and `error="invalid image data"` for index 1.
- `test_non_retryable_and_send_allowance_spent_propagate`: asserts `NonRetryableError` and `SendAllowanceSpent` are re-raised instead of turned into per-image errors.
- `test_single_image_http_400_returns_per_image_error`: asserts HTTP 400 on one image returns `ImageEmbeddingResult(index=0, error=...)` with body detail without crashing sibling images.

In `test_factory.py`:
- Add `test_nvidia_provider` verifying `MultimodalEmbeddingFactory.create(MultimodalProviderConfig(provider="nvidia", api_key="k", model_name="nvidia/nemotron-3.5-embed-8b", dimensions=1024))` returns `NvidiaMultimodalProvider` with `dimensions == 1024` and `base_url == NVIDIA_NIM_BASE_URL`.

In `test_image_providers_rate_limit.py`:
- Add `"nvidia"` to `one_attempt` fixture and add `test_nvidia_hands_the_wait_over` verifying a 429 with `Retry-After: 600` raises `EmbeddingRateLimited(retry_after=600.0)`.

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/unit/services/embeddings/multimodal/test_nvidia_provider.py tests/unit/services/embeddings/multimodal/test_factory.py tests/unit/services/embeddings/test_image_providers_rate_limit.py -v`
Expected: FAIL (`ModuleNotFoundError: nvidia_provider`).

- [ ] **Step 3: Implement `NvidiaMultimodalProvider`, `MultimodalProviderConfig.dimensions`, factory, `VectorStore`, and `health.py` wiring**

1. Add `dimensions: int | None = None` to `MultimodalProviderConfig` in `backend/python/app/services/embeddings/multimodal/config.py`.
2. Create `backend/python/app/services/embeddings/multimodal/nvidia_provider.py`:
   - `_CONCURRENCY_LIMIT = 2`, `_MAX_BATCH_IMAGES = 4`, `_MAX_BATCH_PAYLOAD_BYTES = 4 * 1024 * 1024`, `_REQUEST_TIMEOUT_SECONDS = 60.0`, `_DEFAULT_RATE_LIMIT_DEFER_SECONDS = 60.0`.
   - `_prepare_image_data_uri(normalized_b64: str) -> str | None`: checks `get_mime_type_from_base64(normalized_b64)` (inside try/except for invalid base64); returns `f"data:{mime};base64,{normalized_b64}"` for `"image/png"` / `"image/jpeg"`; otherwise decodes via Pillow (`PIL.Image`), rejects animated images or unreadable bytes, converts to `"RGB"`, encodes as PNG, returns `data:image/png;base64,...`. Run the Pillow conversion path in `asyncio.to_thread`.
   - `_group_image_batches(items: list[tuple[int, str]]) -> list[list[tuple[int, str]]]`: packs up to `_MAX_BATCH_IMAGES` images per batch while total URI length stays within `_MAX_BATCH_PAYLOAD_BYTES` (oversized single images go alone).
   - `NvidiaMultimodalProvider(IMultimodalEmbeddingProvider)`:
     - Resolves `self.base_url = (base_url or "").strip().rstrip("/") or NVIDIA_NIM_BASE_URL` and validates with `require_allowed_endpoint(self.base_url)`.
     - Normalizes `self.dimensions = parse_embedding_dimensions(dimensions)` and gets shared `get_nvidia_rate_limiter(self.base_url, self.api_key)`.
     - Opens one `guarded_async_client(self.base_url, timeout=_REQUEST_TIMEOUT_SECONDS)` and calls `wire.install_wire_hook(client)`.
     - Sends each byte-capped batch to `f"{self.base_url}/embeddings"` through `get_embedding_gateway().call(...)` (acquiring a pacer slot before each send) under `asyncio.Semaphore(_CONCURRENCY_LIMIT)` and `gather_or_cancel`.
     - On multi-image HTTP 400/413/422 failure, falls back to sending that batch's images 1-by-1 so a single corrupt or oversized slide never fails its neighbors.
     - On HTTP 429, calls `pacer.penalize(retry_after or 15.0)`, and if retries exhaust on a 429 (even without a `Retry-After` header), raises `EmbeddingRateLimited(..., retry_after=retry_after or _DEFAULT_RATE_LIMIT_DEFER_SECONDS)`. Re-raises `EmbeddingRateLimited`, `NonRetryableError`, and `refused_by_allowance(exc)`.
     - Maps responses with `map_embedding_response(resp.json().get("data", []), request_indices)`.
3. Register `EmbeddingProvider.NVIDIA.value` in `backend/python/app/services/embeddings/multimodal/factory.py`.
4. In `backend/python/app/modules/transformers/vectorstore.py`:
   - Initialize `self.configured_dimensions: int | None = None` in `__init__` (next to `self.base_url`).
   - Set `self.configured_dimensions = parse_embedding_dimensions(configuration.get("dimensions")) if configuration else None` in `_build_embedding_model`.
   - Pass `dimensions=self.configured_dimensions` in `_multimodal_provider_config()`.
5. In `backend/python/app/api/routes/health.py::_probe_image_embedding`:
   - Pass `dimensions=parse_embedding_dimensions(configuration.get("dimensions"))` into `MultimodalProviderConfig`.

- [ ] **Step 4: Run tests and Ruff to verify Task 3 passes**

Run:
- `pytest tests/unit/services/embeddings/multimodal/test_nvidia_provider.py tests/unit/services/embeddings/multimodal/test_factory.py tests/unit/services/embeddings/test_image_providers_rate_limit.py tests/unit/modules/transformers/test_vectorstore.py -v`
- `ruff check app/services/embeddings/multimodal/nvidia_provider.py app/services/embeddings/multimodal/config.py app/services/embeddings/multimodal/factory.py app/modules/transformers/vectorstore.py app/api/routes/health.py tests/unit/services/embeddings/multimodal/test_nvidia_provider.py tests/unit/services/embeddings/multimodal/test_factory.py tests/unit/services/embeddings/test_image_providers_rate_limit.py`
Expected: PASS, 0 Ruff errors.

- [ ] **Step 5: Stage Task 3 files**

Run: `git add backend/python/app/services/embeddings/multimodal/nvidia_provider.py backend/python/app/services/embeddings/multimodal/config.py backend/python/app/services/embeddings/multimodal/factory.py backend/python/app/modules/transformers/vectorstore.py backend/python/app/api/routes/health.py backend/python/tests/unit/services/embeddings/multimodal/test_nvidia_provider.py backend/python/tests/unit/services/embeddings/multimodal/test_factory.py backend/python/tests/unit/services/embeddings/test_image_providers_rate_limit.py`

---

### Task 4: Health-check error messaging fix (#6333) and NVIDIA vendor reason surfacing

**Files:**
- Modify: `backend/python/app/api/routes/health.py:137-147, 291-340, 480-492`
- Modify: `backend/python/tests/unit/api/routes/test_health_setup_messages.py`
- Modify: `backend/python/tests/unit/api/routes/test_health_failure_reason.py`

**Interfaces:**
- Consumes: `EmbeddingProvider.NVIDIA`, `NvidiaProvider` registered in `ai_model_registry`.
- Produces: Kind-aware `_rejected_settings_message`, NVIDIA vendor error quoting for official `*.api.nvidia.com` hosts only, endpoint-aware `_llm_health_check_needs_outbound` for `nvidia`.

- [ ] **Step 1: Write failing tests in `test_health_setup_messages.py` and `test_health_failure_reason.py`**

In `test_health_setup_messages.py`:
- `test_embedding_rejected_settings_gives_embedding_hint`: calls `_rejected_settings_message("invalid_request", "embedding model", {"provider": "openAICompatible", "configuration": {}}, "nvidia/nemotron-3.5-embed-8b")` and asserts `"output dimensions or multimodal support"` in message and `"reasoning effort"` not in message.
- `test_chat_rejected_settings_keeps_chat_hint`: asserts `"reasoning effort or context length"` stays present for `"chat model"`.
- `test_other_non_chat_rejected_settings_omits_chat_hint`: asserts `"reasoning effort"` not in message for `"image model"` / `"speech model"`.

In `test_health_failure_reason.py`:
- `test_nvidia_official_endpoints_quote_vendor_reason`: checks blank endpoint and `"https://integrate.api.nvidia.com/v1"` quote `NVIDIA NIM said: "..."` on a 400 `openai.BadRequestError` and redact any API key.
- `test_nvidia_custom_self_hosted_endpoint_keeps_server_text_in_log`: checks `"https://nim.internal.example/v1"` and `"https://evil-nvidia.com/v1"` do NOT include `"said:"` in the UI message.
- `test_nvidia_local_endpoint_does_not_require_outbound_internet`: asserts `_llm_health_check_needs_outbound("nvidia", {"endpoint": "http://localhost:8000/v1"}) is False` and `_llm_health_check_needs_outbound("nvidia", {}) is True`.

- [ ] **Step 2: Run health tests to verify they fail**

Run: `pytest tests/unit/api/routes/test_health_setup_messages.py tests/unit/api/routes/test_health_failure_reason.py -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Update `backend/python/app/api/routes/health.py`**

1. In `_llm_health_check_needs_outbound`: include `EmbeddingProvider.NVIDIA.value` in the endpoint-checked tuple.
2. Add `"nvidia"` to `_HOSTED_PROVIDER_KEYS` and define `_NVIDIA_DEFAULT_HOST = "integrate.api.nvidia.com"` and `_NVIDIA_HOST_SUFFIX = ".api.nvidia.com"`.
3. In `_answers_from_vendor`, add `if provider_key == "nvidia":` parsing hostname and returning `host == _NVIDIA_DEFAULT_HOST or host.endswith(_NVIDIA_HOST_SUFFIX)`.
4. In `_rejected_settings_message`, branch on `kind`:
   - `"chat model"`: keep `(such as reasoning effort or context length)`.
   - `"embedding model"`: `(such as output dimensions or multimodal support)`.
   - anything else: `f"{provider} refused these {kind} settings. Check the model name and its options, then try again."`.

- [ ] **Step 4: Run health tests and Ruff to verify Task 4 passes**

Run:
- `pytest tests/unit/api/routes/test_health_setup_messages.py tests/unit/api/routes/test_health_failure_reason.py tests/unit/api/routes/test_health_capabilities.py -v`
- `ruff check app/api/routes/health.py tests/unit/api/routes/test_health_setup_messages.py tests/unit/api/routes/test_health_failure_reason.py`
Expected: PASS, 0 Ruff errors.

- [ ] **Step 5: Stage Task 4 files**

Run: `git add backend/python/app/api/routes/health.py backend/python/tests/unit/api/routes/test_health_setup_messages.py backend/python/tests/unit/api/routes/test_health_failure_reason.py`

---

### Task 5: Frontend icon and provider metadata

**Files:**
- Create: `frontend/public/icons/ai-models/nvidia.svg`
- Modify: `frontend/lib/utils/llm-provider-icons.ts:11-35`
- Modify: `frontend/app/(main)/chat/constants.ts:9-35`
- Modify: `frontend/app/(main)/onboarding/types.ts:76-94`

**Interfaces:**
- Consumes: Provider ID `"nvidia"` and icon path `"/icons/ai-models/nvidia.svg"` from Task 2.
- Produces: Static icon file, icon resolver mapping, friendly display name `"NVIDIA NIM"`, TypeScript `EmbeddingProviderType` member `'nvidia'`.

- [ ] **Step 1: Add NVIDIA SVG icon and update TypeScript mappings**

1. Create `frontend/public/icons/ai-models/nvidia.svg` (clean `viewBox="0 0 24 24"`, `width="1em"`, `height="1em"`, NVIDIA green `#76B900` mark matching the conventions of `frontend/public/icons/ai-models/*.svg`).
2. Add `nvidia: 'nvidia.svg',` to `LLM_PROVIDER_ICON_FILE` in `frontend/lib/utils/llm-provider-icons.ts`.
3. Add `nvidia: 'NVIDIA NIM',` to `PROVIDER_FRIENDLY_NAMES` in `frontend/app/(main)/chat/constants.ts`.
4. Add `| 'nvidia'` to `EmbeddingProviderType` in `frontend/app/(main)/onboarding/types.ts`.

- [ ] **Step 2: Verify frontend metadata and SVG validity**

Verify:
- `frontend/public/icons/ai-models/nvidia.svg` parses as valid XML/SVG.
- If `frontend/node_modules` is present (or check TypeScript syntax / icon test via node or vitest), verify `resolveLlmProviderIconPath('nvidia') === '/icons/ai-models/nvidia.svg'` and `PROVIDER_FRIENDLY_NAMES.nvidia === 'NVIDIA NIM'`.

- [ ] **Step 3: Stage Task 5 files**

Run: `git add frontend/public/icons/ai-models/nvidia.svg frontend/lib/utils/llm-provider-icons.ts frontend/app/(main)/chat/constants.ts frontend/app/(main)/onboarding/types.ts`

---

### Task 6: GitHub Actions ARM64 Docker workflow, zero-data-loss OCI VPS upgrade runbook, and full verification

**Files:**
- Create: `.github/workflows/fork-docker-arm64.yml`
- Create: `docs/superpowers/runbooks/oci-arm64-fork-upgrade.md`

**Interfaces:**
- Consumes: Existing root `Dockerfile`, upstream prebuilt base images (`ghcr.io/pipeshub-ai/pipeshub-ai-base:python-deps-slim` and `runtime`, Docker Hub fallback), repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, and `deployment/docker-compose/{docker-compose.yml,install.sh}`.
- Produces: Manually triggered ARM64 slim build-and-push workflow and operator upgrade/rollback runbook.

- [ ] **Step 1: Write `.github/workflows/fork-docker-arm64.yml`**

Implement the `workflow_dispatch`-only workflow specified in Spec Component 6:
- `inputs.tag` (default `"slim"`, validated against safe Docker tag regex `^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$` so a bad input fails cleanly before building).
- `inputs.push_sha_tag` (boolean, default `true`).
- `runs-on: ubuntu-24.04-arm`, `timeout-minutes: 60`, `permissions: { contents: read }`.
- Reclaim disk space on runner, confirm `dpkg --print-architecture` is `arm64`, resolve `PYTHON_DEPS_IMAGE` and `RUNTIME_BASE_IMAGE` (GHCR with Docker Hub fallback), fail fast if `DOCKERHUB_USERNAME` or `DOCKERHUB_TOKEN` secret is empty, build and push `linux/arm64` via `docker/build-push-action` with GHA cache, and write pushed tags + digest to `$GITHUB_STEP_SUMMARY`.

- [ ] **Step 2: Write `docs/superpowers/runbooks/oci-arm64-fork-upgrade.md`**

Include copy-pasteable commands for:
1. Read-only pre-upgrade server check (`docker inspect` working dir, project name, current image digest, volume list, `PIPESHUB_IMAGE` check in `docker-compose.yml`).
2. Mandatory `.env` (`chmod 600`) and compose backups (never `down -v`, never regenerate `.env`).
3. Refreshing `docker-compose.yml` + `install.sh` from `zenwav/pipeshub-ai` only if the server's compose file predates `PIPESHUB_IMAGE`.
4. Setting `PIPESHUB_IMAGE=<dockerhub-username>/pipeshub-ai` while keeping `IMAGE_TAG=slim` (plus `SANDBOX_DOCKER_IMAGE` note when pinning SHA tags).
5. Running `bash ./install.sh --upgrade` (and app-only alternative `docker compose pull pipeshub-ai && docker compose up -d --no-deps pipeshub-ai`), verifying `/api/v1/health/services`, sign-in, and preserved keys/models.
6. Configuring **NVIDIA NIM** (`nvidia/nemotron-3.5-embed-8b`, Multimodal on, native 4096 dims) and clearing/reindexing Qdrant embeddings if documents were already indexed under BGE.
7. Instant rollback steps.

- [ ] **Step 3: Validate workflow YAML, run full Python test + coverage + Ruff suite, and verify `git diff --cached`**

Run:
- Parse `.github/workflows/fork-docker-arm64.yml` with PyYAML (`yaml.safe_load`) to guarantee valid YAML syntax.
- Run all unit tests + coverage on all new and modified Python modules:
  `pytest --cov=app.utils.nvidia_embeddings --cov=app.services.embeddings.multimodal.nvidia_provider --cov=app.config.ai_models.providers.nvidia --cov-report=term-missing tests/unit/utils/test_nvidia_embeddings.py tests/unit/services/embeddings/multimodal/test_nvidia_provider.py tests/unit/services/embeddings/multimodal/test_factory.py tests/unit/services/embeddings/test_image_providers_rate_limit.py tests/unit/api/routes/test_health_setup_messages.py tests/unit/api/routes/test_health_failure_reason.py tests/unit/config/ai_models/test_ai_model_providers.py tests/unit/utils/test_ai_model_factory_contract.py tests/unit/services/ai_models/test_discovery.py tests/unit/services/load/test_projection.py tests/unit/services/model_identity/test_provider_coverage.py tests/unit/utils/test_aimodels.py`
- Run the offline wire-hook contract test for `nvidia`:
  `pytest tests/integration/spend/test_wire_contract.py::test_every_embedding_provider_builds_with_the_wire_hooks -k nvidia -v`
- Run `ruff check` on every created and modified Python file.
- Run `git status -sb` and `git diff --cached --stat` to confirm everything is cleanly staged, nothing is committed, nothing is pushed, and no stray temporary files exist.

- [ ] **Step 4: Stage Task 6 files and deliver summary + Claude Code review prompt**

Run: `git add .github/workflows/fork-docker-arm64.yml docs/superpowers/runbooks/oci-arm64-fork-upgrade.md docs/superpowers/plans/2026-11-24-nvidia-nim-embedding-provider.md`
(Do NOT commit or push; leave staged for Carbon and Claude Code review.)
