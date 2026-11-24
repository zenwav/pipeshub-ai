# Native NVIDIA NIM multimodal embedding provider and ARM64 fork deployment

Date: 2026-11-24
Issue: https://github.com/pipeshub-ai/pipeshub-ai/issues/6333
Fork: `zenwav/pipeshub-ai`

> **Revised 2026-11-25 (review fixes).** Where this document and the code differ, the code is authoritative:
> - Pacing moved to `app/utils/nvidia_pacer.py`: one state file per endpoint and key, shared by every process in the container under `flock` (the query and indexing services are separate processes, so a per-process limiter let each spend the whole allowance). Defaults: 36 requests a minute in total on NVIDIA Build, of which indexing may use 30, leaving 6 for searches (`NVIDIA_NIM_RPM`, `NVIDIA_NIM_INTERACTIVE_RESERVE_RPM`). Nothing is booked ahead, so a cancelled caller leaves nothing behind; a wait longer than 45 s hands the work back as `EmbeddingRateLimited` instead of sleeping past the batch deadline.
> - A 429 without `Retry-After` on document indexing raises `EmbeddingRateLimited` (record deferred 60 s) instead of writing a fake `Retry-After` header into NVIDIA's response. Searches keep the caller's retries.
> - Response vectors are refused when their indexes are repeated or out of range.
> - The provider reuses the shared embedding fields (`EMBEDDING_COMMON_TAIL`): Multimodal defaults to off, as for every other embedding provider. The icon is `nvidia-color.svg`.
> - Decision-model rejections keep the LLM hint (reasoning effort, context length).

## Problem and goals

Adding `nvidia/nemotron-3.5-embed-8b` through **Workspace settings > AI models > For embedding > OpenAI Compatible** fails health check with HTTP 400 (`'input_type' parameter is required for asymmetric models`) because `OpenAIEmbeddings` does not send NVIDIA's required `input_type` parameter (`passage` for document and image indexing, `query` for search queries). The modal also displays an LLM-only error hint (`such as reasoning effort or context length`) on embedding failures.

This design solves four concrete goals:
1. Add a native **NVIDIA NIM** embedding provider (`nvidia`) scoped to embedding and multimodal image indexing for `nvidia/nemotron-3.5-embed-8b`, with strict accuracy guarantees for lecture slides, textbooks, diagrams, and text chunks.
2. Fix embedding health-check error messages so embedding failures give embedding-relevant advice and official NVIDIA NIM API errors surface cleanly in the UI without relaxing PipesHub's self-hosted error-isolation policy.
3. Add a manual GitHub Actions workflow (`workflow_dispatch`) on `zenwav/pipeshub-ai` that builds and pushes a native `linux/arm64` `slim` Docker image to Docker Hub on GitHub's free `ubuntu-24.04-arm` runner.
4. Provide a repeatable, zero-data-loss upgrade and rollback procedure for an existing standalone `curl -fsSL https://get.pipeshub.com/install | bash` (`slim`) deployment running behind Nginx on OCI Ubuntu 26.04 (4 ARM Ampere A1 cores).

## Non-goals

- No NVIDIA NIM LLM (text generation), TTS, STT, image generation, or decision model capability.
- No new Python third-party packages (no `langchain-nvidia-ai-endpoints`); reuse `openai==2.45.0`, `httpx==0.28.1`, and `Pillow` already baked into `pipeshubai/pipeshub-ai-base:python-deps-slim`.
- No edits to upstream `.github/workflows/release-publish.yml` so future upstream rebases stay conflict-free.
- No automatic CI builds on git push.
- No `git push` until the user finishes their own review and explicitly approves pushing.

## Architecture overview

The native `nvidia` provider follows PipesHub's existing provider registry, factory, egress guard, identity forwarding, and wire-hook contracts:

```text
Dashboard UI (dynamic form from /api/v1/ai-models/registry)
  -> Node Configuration Manager (stores encrypted config in KV)
  -> Python Query / Indexing services
       |- Text indexing  : NvidiaEmbeddings.aembed_documents -> POST /v1/embeddings (input_type="passage", truncate="END")
       |- Text search    : NvidiaEmbeddings.aembed_query     -> POST /v1/embeddings (input_type="query",   truncate="END")
       |- Image indexing : NvidiaMultimodalProvider          -> POST /v1/embeddings (input_type="passage", modality="image", truncate="END")
       |- Health check   : perform_embedding_health_check    -> text probe + image probe + dimension match + Qdrant guard
```

## Component 1: Provider registration and model discovery

### Registry declaration
Create `backend/python/app/config/ai_models/providers/nvidia.py` and register `NvidiaProvider` in `ALL_PROVIDER_CLASSES` inside `backend/python/app/config/ai_models/providers/__init__.py`.

Provider metadata:
- Display name: `"NVIDIA NIM"`
- Provider ID: `"nvidia"`
- Description: `"NVIDIA NIM multimodal and asymmetric embedding models"`
- Capabilities: `[ModelCapability.EMBEDDING]`
- Icon path: `"/icons/ai-models/nvidia-color.svg"`
- Color: `"#76B900"`

Fields for `ModelCapability.EMBEDDING`:
1. `apiKey`: shared `API_KEY` (`PASSWORD`, required, secret).
2. `model`: `AIModelField(name="model", display_name="Model Name", field_type="TEXT", required=True, default_value="nvidia/nemotron-3.5-embed-8b", placeholder="e.g., nvidia/nemotron-3.5-embed-8b")`.
3. `endpoint`: `AIModelField(name="endpoint", display_name="Endpoint URL (Optional)", field_type="URL", required=False, placeholder="https://integrate.api.nvidia.com/v1", description="Leave blank for NVIDIA Build (integrate.api.nvidia.com/v1), or enter a self-hosted NIM endpoint.")`.
4. Shared tail (`EMBEDDING_COMMON_TAIL`): `modelFriendlyName` (`FRIENDLY_NAME`), `dimensions` (`DIMENSIONS`, optional Matryoshka truncation, 1 to 4096 for `nvidia/nemotron-3.5-embed-8b`), and `isMultimodal` (`MULTIMODAL`, `default_value=False`; turn on in the UI to index slide and document images directly into the same vector space as text).

### Model discovery
In `backend/python/app/services/ai_models/discovery/strategies.py`, register `"nvidia"` in `build_registry()` using `_StaticStrategy`:

```python
_NVIDIA = [
    RawModel(
        id="nvidia/nemotron-3.5-embed-8b",
        capabilities=["embedding"],
        is_multimodal=True,
    ),
]
```

This satisfies `test_every_registered_provider_has_a_discovery_strategy`, pre-populates `nvidia/nemotron-3.5-embed-8b` with multimodal enabled in the picker, and avoids listing unrelated NVIDIA chat/rerank models.

## Component 2: Text embedding client (`NvidiaEmbeddings`)

### Constants and provider enum
- In `backend/python/app/config/constants/ai_models.py`, define:
  `NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"`
- In `backend/python/app/utils/aimodels.py`:
  - Add `NVIDIA = "nvidia"` to `EmbeddingProvider`.
  - Add `EmbeddingProvider.NVIDIA.value` to `SELF_HOSTABLE_EMBEDDING_PROVIDERS` so custom private NIM endpoints get local timeout sizing while the default public `integrate.api.nvidia.com` host gets remote timeout sizing (`60.0s`).

### Client implementation
Create `backend/python/app/utils/nvidia_embeddings.py` implementing `NvidiaEmbeddings(BaseModel, Embeddings)`:

Fields:
- `client: Any = Field(default=None, exclude=True)` (holds `openai.OpenAI(...).embeddings`)
- `async_client: Any = Field(default=None, exclude=True)` (holds `openai.AsyncOpenAI(...).embeddings`)
- `model: str = "nvidia/nemotron-3.5-embed-8b"`
- `nvidia_api_key: SecretStr | None = Field(alias="api_key", default=None)`
- `nvidia_api_base: str = Field(default=NVIDIA_NIM_BASE_URL, alias="base_url")`
- `dimensions: int | None = None`
- `truncate: Literal["NONE", "END"] = "END"`
- `batch_size: int = 50`
- `max_retries: int = 0`
- `request_timeout: float | tuple[float, float] | None = Field(default=None, alias="timeout")`

Client construction in `@model_validator(mode="after")`:
- Builds `openai.OpenAI(**client_params).embeddings` and `openai.AsyncOpenAI(**client_params).embeddings`.
- Because `client._client` and `async_client._client` resolve to `openai.OpenAI` and `openai.AsyncOpenAI` (which hold `httpx.Client` and `httpx.AsyncClient` on `._client`), PipesHub's existing `OPENAI_RESOURCE_CLIENTS = ("client._client", "async_client._client")` path table automatically hooks:
  - `_guard_clients` (SSRF protection and redirect refusal),
  - `instrument_model` (identity forwarding and `test_provider_coverage.py`),
  - `llm_gateway/wire.py` (spend guard, fleet breaker, and `test_wire_contract.py`).

Request and alignment guarantees:
- Empty `texts` list in `embed_documents` / `aembed_documents` returns `[]` without making an HTTP call.
- Batches `texts` in slices of `self.batch_size` (default 50).
- Builds request kwargs per call:
  - `model=self.model`
  - `input=batch` (raw strings, never tiktoken token-ID arrays)
  - `encoding_format="float"`
  - `extra_body={"input_type": input_type, "truncate": self.truncate}`
  - `dimensions=self.dimensions` only when `self.dimensions is not None`
- `embed_documents` and `aembed_documents` always send `input_type="passage"`.
- `embed_query` and `aembed_query` always send `input_type="query"` with `input=[text]`.
- Response extraction sorts items by `item.index` (falling back to positional index only if `index` is absent) and verifies `len(vectors) == len(batch)` and every vector is a non-empty list of floats. Any mismatch raises `ValueError` instead of silently shifting chunk-to-vector alignment in Qdrant.

### Rate limiting for NVIDIA Build free-tier keys (40 RPM cap)
Hosted NVIDIA Build (`integrate.api.nvidia.com`) allows 40 requests a minute per free key, and its `429 Too Many Requests` responses often omit `Retry-After`. The query service (searches, model checks) and the indexing service (documents, images) are separate processes sending with the same key, so pacing must be shared between processes.

`app/utils/nvidia_pacer.py` keeps one small JSON state file per endpoint and key in the directory the container's processes already share for resource slots (`shared_slots_directory("nvidia-nim-pacer")`, created private by `ensure_private_directory`), read and rewritten under an exclusive `flock`. Without POSIX locks or a safe directory each process keeps its own state, with one warning.
- **Two lanes:** document and image indexing (`BULK`) may make at most `total - reserve` of the sends in any 60 s window and is spaced evenly (`60 / bulk` s); searches (`INTERACTIVE`) may use the whole `total`, so they are not queued behind a document backlog.
- **Defaults:** NVIDIA Build: `total` 36 (4 below NVIDIA's 40 for window skew), `reserve` 6, so indexing runs at 30 a minute. Self-hosted NIM: pacing off.
- **Configuration:** `NVIDIA_NIM_RPM` (total; `0` turns pacing off) and `NVIDIA_NIM_INTERACTIVE_RESERVE_RPM`, both passed through in `deployment/docker-compose/docker-compose.yml`.
- **No booking ahead:** a caller that cannot send sleeps until the earliest moment it could and asks again, so a cancelled caller leaves nothing behind. A wait over 45 s raises `PacerBusy`, which the providers turn into `EmbeddingRateLimited` (the record is handed back) rather than sleeping past the 120 s batch deadline.
- **429 cooldown:** a 429 sets a cooldown for every lane of the key (`Retry-After`, or 15 s without one). A document batch whose 429 named no wait is handed back as `EmbeddingRateLimited(retry_after=60)`; a search keeps the caller's own retries.

### Factory wiring
In `backend/python/app/utils/aimodels.py::_get_embedding_model_impl`:
- Add `elif provider == EmbeddingProvider.NVIDIA.value:` branch.
- Resolve `endpoint = (configuration.get("endpoint") or "").strip() or NVIDIA_NIM_BASE_URL`.
- Call `require_allowed_endpoint(endpoint)` (already runs on `configuration.get("endpoint")` at top of function; also validate resolved URL).
- Apply `_set_embedding_dimensions_kwarg(nvidia_kwargs, dimensions)` and `_set_openai_client_limits_kwargs(nvidia_kwargs, provider, endpoint)`.
- Return `_guard_clients(NvidiaEmbeddings(**nvidia_kwargs), _OPENAI_RESOURCE_CLIENTS, endpoint)`.

Note on `embedding_config_hash`: `provider`, `isDefault`, `isMultimodal`, `model`, `endpoint`, `dimensions`, and `apiKey` are already hashed in `embedding_config_hash`, so any config edit invalidates cached clients cleanly.

## Component 3: Multimodal image embedding provider (`NvidiaMultimodalProvider`)

### Propagating optional `dimensions` to multimodal providers
- Add `dimensions: int | None = None` to `MultimodalProviderConfig` in `backend/python/app/services/embeddings/multimodal/config.py` (defaults to `None` so all existing callers stay compatible).
- In `VectorStore._build_embedding_model` (`backend/python/app/modules/transformers/vectorstore.py`), store parsed positive integer `self.configured_dimensions` from `configuration.get("dimensions")` and pass `dimensions=self.configured_dimensions` in `_multimodal_provider_config()`.
- In `health.py::_probe_image_embedding`, parse optional positive integer `dimensions` from `configuration.get("dimensions")` and pass `dimensions=requested_dimensions` into `MultimodalProviderConfig`.

Why a separate `dimensions` field instead of always sending `embedding_size`: when a user leaves Output Dimensions blank, both text and image calls omit `dimensions` and use the model's native 4096 dimensions. When a user explicitly sets Output Dimensions (for example `1024` or `2048`), both text and image calls send that exact `dimensions` value so Matryoshka vectors match across text and images. Both `health.py` and `VectorStore._build_image_points` still assert `len(image_vector) == text_dimension` as a hard guard.

### Provider implementation
Create `backend/python/app/services/embeddings/multimodal/nvidia_provider.py` implementing `NvidiaMultimodalProvider(IMultimodalEmbeddingProvider)` and register it in `backend/python/app/services/embeddings/multimodal/factory.py` under `EmbeddingProvider.NVIDIA.value`.

Parameters:
- `api_key: str | None`
- `model_name: str | None`
- `base_url: str | None = None` (falls back to `NVIDIA_NIM_BASE_URL` when blank/None)
- `dimensions: int | None = None`
- `normalize_fn: Callable[[str], Any] | None = None`
- `logger: logging.Logger | None = None`

Image normalization and MIME accuracy:
1. Validate and strip any existing data URI via `normalized = await self.normalize(image_ref)`. If `None`, return `ImageEmbeddingResult(index=i, error="invalid image data")`.
2. Detect true image format from magic bytes using `get_mime_type_from_base64(normalized)` from `app/utils/image_utils.py`.
3. If `mime in ("image/png", "image/jpeg")`, build `f"data:{mime};base64,{normalized}"`.
4. Otherwise (WebP, GIF, BMP, TIFF, or unrecognized header), decode via Pillow in `asyncio.to_thread`, convert to `RGB`, save as lossless PNG into a `BytesIO` buffer, and base64-encode as `data:image/png;base64,...`. Lossless PNG preserves sharp text, formulas, and plots on lecture slides. If Pillow cannot decode the bytes, return `ImageEmbeddingResult(index=i, error="invalid image data")`.

Request execution and reliability:
- **Byte-capped image batching with 1-by-1 isolation fallback:** Groups up to **4 images** per HTTP POST when combined base64 payload stays within **4 MiB** (a single image larger than 4 MiB goes in its own 1-image request), running under `asyncio.Semaphore(2)` concurrency and sharing `NvidiaRateLimiter` with text calls. That cuts RPM usage up to 4x on lecture slide decks (100 slides takes ~25 RPM requests, under 45 seconds at 35 RPM). If a multi-image batch fails with HTTP 400, 413, or 422 (for example one oversized or corrupt slide), the provider automatically retries those images 1-by-1 so sibling slides still index cleanly.
- Uses `guarded_async_client(self.base_url, timeout=60.0)` with `wire.install_wire_hook(client)` so SSRF policy, redirect refusal, connect-timeout cap, send-allowance metering, and fleet-breaker telemetry all apply.
- JSON body:
  ```json
  {
    "model": "<model_name>",
    "input": ["data:image/png;base64,..."],
    "encoding_format": "float",
    "input_type": "passage",
    "modality": "image",
    "truncate": "END"
  }
  ```
  Adds `"dimensions": self.dimensions` when `self.dimensions` is a positive integer.
- Calls `get_embedding_gateway().call(...)` and runs tasks through `gather_or_cancel`.
- Re-raises `EmbeddingRateLimited`, `NonRetryableError`, and any exception where `refused_by_allowance(e)` is true. When NVIDIA returns HTTP 429 without a `Retry-After` header and retries exhaust, converts the 429 into `EmbeddingRateLimited(..., retry_after=60.0)` and penalizes the shared pacer so PipesHub automatically defers and retries the record with zero manual re-indexing.
- Maps HTTP responses with `map_embedding_response(data, request_indices)` and logs per-image errors with `describe_request_error(e)`.

### Factory update
In `backend/python/app/services/embeddings/multimodal/factory.py`:
- Register `EmbeddingProvider.NVIDIA.value` building `NvidiaMultimodalProvider(api_key=config.api_key, model_name=config.model_name, base_url=config.base_url, dimensions=config.dimensions, normalize_fn=config.normalize_fn, logger=config.logger)`.
- Note: `MultimodalEmbeddingFactory.create` calls `require_allowed_endpoint(config.base_url)`; when `config.base_url` is `None` (default NVIDIA endpoint), `require_allowed_endpoint(None)` is a no-op and `NvidiaMultimodalProvider` resolves `NVIDIA_NIM_BASE_URL` and validates it with `require_allowed_endpoint`.

## Component 4: Health checks and error messaging fix (#6333)

In `backend/python/app/api/routes/health.py`:
1. **`_rejected_settings_message`:**
   - When `kind == "chat model"`, preserve the existing wording:
     `f"{provider} refused these {kind} settings. Check the model name and its options (such as reasoning effort or context length), then try again."`
   - When `kind == "embedding model"`, return embedding-specific guidance:
     `f"{provider} refused these {kind} settings. Check the model name and its options (such as output dimensions or multimodal support), then try again."`
   - For `chat model` and `decision model`, keep the LLM hint (`such as reasoning effort or context length`).
   - For other non-chat kinds (`image model`, `speech model`, `speech-to-text model`), omit chat-specific examples:
     `f"{provider} refused these {kind} settings. Check the model name and its options, then try again."`
2. **Hosted vendor reason surfacing:**
   - Add `"nvidia"` to `_HOSTED_PROVIDER_KEYS`.
   - Define `_NVIDIA_HOST_SUFFIX = ".api.nvidia.com"` and `_NVIDIA_DEFAULT_HOST = "integrate.api.nvidia.com"`.
   - In `_answers_from_vendor`, add a `"nvidia"` branch that parses the configured `endpoint` host and returns `True` only when `host == _NVIDIA_DEFAULT_HOST or host.endswith(_NVIDIA_HOST_SUFFIX)`. Blank endpoints already return `True`. Self-hosted NIM endpoints keep raw server text in logs only, preserving PipesHub's existing security boundary tested by `test_health_failure_reason.py`.
3. **Outbound connectivity check:**
   - In `_llm_health_check_needs_outbound`, include `EmbeddingProvider.NVIDIA.value` alongside `openAICompatible` and `litellmProxy` so a custom private/local NIM endpoint is not treated as requiring outbound internet access. (When `endpoint` is blank, `_endpoint_is_local("")` returns `False`, so default cloud NIM still checks outbound connectivity.)

## Component 5: Cross-cutting invariants and frontend metadata

1. **Resource governor dependency normalization (`backend/python/app/services/resource_governor/explain.py`):**
   - Add `"nvidia"` to `EMBEDDING_PROVIDER_NAMES` (strictly pinned to `{v.value.lower() for v in EmbeddingProvider}` by `tests/unit/services/load/test_projection.py`).
2. **Identity forwarding coverage (`backend/python/tests/unit/services/model_identity/test_provider_coverage.py`):**
   - Add `"nvidia"` to the parametrized list in `test_embedding_clients_carry_the_hook` and to `known` in `test_every_embedding_provider_is_classified`.
3. **Frontend icon and display metadata:**
   - Add `frontend/public/icons/ai-models/nvidia-color.svg` (clean minimal NVIDIA brand mark SVG).
   - Map `nvidia: 'nvidia-color.svg'` in `frontend/lib/utils/llm-provider-icons.ts`.
   - Map `nvidia: 'NVIDIA NIM'` in `PROVIDER_FRIENDLY_NAMES` inside `frontend/app/(main)/chat/constants.ts`.
   - Add `'nvidia'` to `EmbeddingProviderType` in `frontend/app/(main)/onboarding/types.ts`.
   - Note: `frontend/app/(main)/workspace/ai-models/` renders providers dynamically from `GET /api/v1/ai-models/registry`, so the new **NVIDIA NIM** card under **For embedding** appears automatically with zero hardcoded dialog changes.

## Component 6: GitHub Actions ARM64 Docker publish workflow

Create `.github/workflows/fork-docker-arm64.yml` (standalone file; leaves upstream `release-publish.yml` untouched).

- **Trigger:** `workflow_dispatch` only.
  - `tag` (string, default `"slim"`): primary Docker Hub tag to publish.
  - `push_sha_tag` (boolean, default `true`): also publish immutable `slim-<7-char-sha>` tag for rollback pinning.
- **Permissions:** `contents: read`.
- **Concurrency:** `group: fork-docker-arm64-${{ github.ref }}`, `cancel-in-progress: false`.
- **Runner:** `ubuntu-24.04-arm` (`timeout-minutes: 60`).
- **Steps:**
  1. Checkout repository (`actions/checkout`).
  2. Verify runner architecture is `arm64` (`dpkg --print-architecture`).
  3. Reclaim runner disk space (`/usr/share/dotnet`, `/usr/local/lib/android`, `/opt/ghc`, `/usr/local/share/boost`, `${AGENT_TOOLSDIRECTORY:-}`).
  4. Resolve prebuilt upstream base images (`ghcr.io/pipeshub-ai/pipeshub-ai-base:python-deps-slim` and `runtime`, falling back to `pipeshubai/pipeshub-ai-base:*` if GHCR pull fails).
  5. Set up Docker Buildx (`docker/setup-buildx-action`).
  6. Validate that `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are configured (fail early with a clear message if missing), then log in via `docker/login-action`.
  7. Compute short SHA and tag list (`${DOCKERHUB_USERNAME}/pipeshub-ai:${tag}`, plus `${DOCKERHUB_USERNAME}/pipeshub-ai:${tag}-${short_sha}` when `push_sha_tag` is true).
  8. Build and push `linux/arm64` with `docker/build-push-action`:
     - `context: .`, `file: Dockerfile`, `platforms: linux/arm64`, `push: true`
     - `build-args`: `PYTHON_DEPS_IMAGE`, `RUNTIME_BASE_IMAGE`, `APP_VERSION=fork-${short_sha}`, `GIT_COMMIT=${{ github.sha }}`
     - `cache-from: type=gha,scope=fork-arm64-slim` and `cache-to: type=gha,scope=fork-arm64-slim,mode=max`
  9. Print the pushed image tags and digest to `$GITHUB_STEP_SUMMARY`.

Required one-time GitHub repository secrets on `zenwav/pipeshub-ai`:
- `DOCKERHUB_USERNAME`: Docker Hub username.
- `DOCKERHUB_TOKEN`: Docker Hub Personal Access Token with Read and Write permissions.

## Component 7: Zero-data-loss OCI Ubuntu 26.04 VPS upgrade guide

Documented in `docs/superpowers/runbooks/oci-arm64-fork-upgrade.md` (and summarized in the final handoff):

1. **Pre-upgrade server check (read-only):**
   - Locate the live compose working directory and project name via `docker inspect` on the running `pipeshub-ai` container (`com.docker.compose.project.working_dir`, `com.docker.compose.project`).
   - Record the current running image and digest (`docker inspect --format='{{.Config.Image}} {{index .RepoDigests 0}}'`).
   - Confirm named volumes (`mongodb_data`, `neo4j_data`, `redis_data`, `qdrant_storage`, `pipeshub_data`, `pipeshub_root_local`, `pipeshub_hf_cache`) and check whether `docker-compose.yml` already references `${PIPESHUB_IMAGE:-pipeshubai/pipeshub-ai}`.
2. **Backup before changing anything:**
   - Copy `.env` to `.env.backup.$(date +%Y%m%d-%H%M%S)` with `chmod 600` (`.env` holds `SECRET_KEY`, which decrypts every saved API key and model config, plus DB passwords).
   - Copy `docker-compose.yml` and `install.sh` to `.bak` files before updating them (if the existing `docker-compose.yml` predates `PIPESHUB_IMAGE` support).
   - Never run `docker compose down -v` and never regenerate `.env`.
3. **Switch image source to the fork:**
   - If the server's `docker-compose.yml` does not contain `PIPESHUB_IMAGE`, download the fork's `deployment/docker-compose/docker-compose.yml` and `deployment/docker-compose/install.sh` into the install directory (keeping `.env` untouched).
   - Add `PIPESHUB_IMAGE=<dockerhub-username>/pipeshub-ai` to `.env` while keeping `IMAGE_TAG=slim` (so `SANDBOX_DOCKER_IMAGE` continues to resolve to `pipeshubai/pipeshub-sandbox:slim`). If pinning `IMAGE_TAG=slim-<sha>`, also set `SANDBOX_DOCKER_IMAGE=pipeshubai/pipeshub-sandbox:slim`.
4. **Upgrade in place:**
   - Run `bash ./install.sh --upgrade` (or `docker compose pull pipeshub-ai && docker compose up -d --no-deps pipeshub-ai`).
   - Existing Nginx reverse proxy config, TLS certs, public URL, user accounts, encrypted API keys, and configured LLMs remain untouched.
   - Verify `/api/v1/health/services` and confirm sign-in and existing configured models in the browser.
5. **Configure NVIDIA NIM and reindex accurately:**
   - Open **Workspace settings > AI models > For embedding > NVIDIA NIM**.
   - Enter NVIDIA API key, keep model `nvidia/nemotron-3.5-embed-8b`, leave Endpoint URL blank, keep **Multimodal** checked, and leave **Output Dimensions** blank for native 4096 dimensions (20,000 chunks at 4096 float32 dimensions is roughly 328 MB, comfortably inside Qdrant's default 6 GB limit).
   - If documents were already indexed under `BAAI/bge-large-en-v1.5`, run **Clean up embeddings** and **Reindex embeddings** so Qdrant holds only vectors from `nvidia/nemotron-3.5-embed-8b` and never mixes incompatible vector spaces.
6. **Rollback procedure:**
   - Restore `.env` from `.env.backup.*` (removing `PIPESHUB_IMAGE`) and run `bash ./install.sh --upgrade` to return to upstream `pipeshubai/pipeshub-ai:slim` with all volumes and secrets preserved.

## Verification and testing plan

New unit tests:
- `backend/python/tests/unit/utils/test_nvidia_embeddings.py`:
  - Sync and async `embed_documents` / `aembed_documents` pass `input_type="passage"`, `truncate="END"`, `encoding_format="float"`.
  - Sync and async `embed_query` / `aembed_query` pass `input_type="query"`, `truncate="END"`, `encoding_format="float"`.
  - Optional `dimensions` omitted when unset and forwarded when configured.
  - Empty `texts` returns `[]` without network calls; >50 texts splits into 50-item batches and concatenates in order.
  - Response items returned out of order are re-sorted by `index`; partial/short responses raise `ValueError`.
  - Factory `get_embedding_model("nvidia", ...)` defaults to `NVIDIA_NIM_BASE_URL`, honors custom `endpoint`, blocks private/metadata endpoints per egress policy, sets `max_retries=0`, and installs httpx hooks.
- `backend/python/tests/unit/services/embeddings/multimodal/test_nvidia_provider.py`:
  - PNG and JPEG images preserve exact magic-byte MIME types in `data:image/...;base64,...`.
  - Non-PNG/JPEG formats (WebP, BMP, GIF) convert cleanly via Pillow to RGB PNG data URIs.
  - Invalid/corrupt base64 returns `ImageEmbeddingResult(index=i, error="invalid image data")` without aborting sibling images.
  - Sends 1 image per HTTP call with `input_type="passage"`, `modality="image"`, `truncate="END"`, `encoding_format="float"`, and forwards `dimensions` only when explicitly set.
  - Propagates `EmbeddingRateLimited`, `NonRetryableError`, and `SendAllowanceSpent` (rate-limit deferral cancels sibling tasks cleanly).
  - `MultimodalEmbeddingFactory.create` instantiates `NvidiaMultimodalProvider` and passes `dimensions`.

Updated existing invariant and regression tests:
- `backend/python/tests/unit/api/routes/test_health_setup_messages.py`: verifies embedding refused-settings message mentions output dimensions / multimodal instead of reasoning effort / context length, while chat model wording stays unchanged.
- `backend/python/tests/unit/api/routes/test_health_failure_reason.py`: verifies official NVIDIA NIM (`integrate.api.nvidia.com` and blank endpoint) quotes redacted vendor reason while a custom self-hosted NIM endpoint suppresses raw server text.
- `backend/python/tests/unit/services/model_identity/test_provider_coverage.py`: includes `"nvidia"` in verified httpx embedding providers.
- `backend/python/tests/unit/services/embeddings/test_image_providers_rate_limit.py`: includes `"nvidia"` in rate-limit deferral coverage.
- Existing suites verified green: `test_ai_model_providers.py`, `test_ai_model_factory_contract.py`, `test_discovery.py`, `test_projection.py`, `test_wire_contract.py` (offline wire-hook tests), `test_aimodels.py`, `test_factory.py`, `test_vectorstore.py`, Ruff linter, Python coverage on all new/modified modules, and frontend TypeScript/Vitest checks.

## Approval and push gate

Per user instructions:
- Skip automated subagent review; run unit tests, coverage checks, and linters locally.
- Leave changes committed locally without pushing (`git push` waits until after the user switches to Carbon in the model picker, runs their checks, and gives explicit approval to push).
