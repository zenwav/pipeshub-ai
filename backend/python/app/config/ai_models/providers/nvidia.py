"""NVIDIA NIM embedding provider registration."""

from __future__ import annotations

from app.config.ai_models.registry import AIModelProviderBuilder
from app.config.ai_models.types import AIModelField, ModelCapability
from app.config.constants.ai_models import NVIDIA_NIM_BASE_URL
from app.utils.nvidia_embeddings import DEFAULT_NVIDIA_EMBEDDING_MODEL

from .common_fields import API_KEY, EMBEDDING_COMMON_TAIL

_NVIDIA_MODEL = AIModelField(
    name="model",
    display_name="Model Name",
    field_type="TEXT",
    required=True,
    default_value=DEFAULT_NVIDIA_EMBEDDING_MODEL,
    placeholder=f"e.g., {DEFAULT_NVIDIA_EMBEDDING_MODEL}",
    # Every request names input_type (and images name modality), which only NVIDIA's
    # asymmetric retrieval models accept.
    description=(
        "An NVIDIA asymmetric retrieval embedding model that takes input_type "
        "(passage for documents, query for searches)."
    ),
)

_NVIDIA_ENDPOINT = AIModelField(
    name="endpoint",
    display_name="Endpoint URL (Optional)",
    field_type="URL",
    required=False,
    placeholder=NVIDIA_NIM_BASE_URL,
    description=(
        "Leave blank for NVIDIA Build (integrate.api.nvidia.com/v1), "
        "or enter a self-hosted NIM endpoint."
    ),
)


@AIModelProviderBuilder("NVIDIA NIM", "nvidia") \
    .with_description("NVIDIA NIM asymmetric retrieval embeddings (input_type passage/query), text and images") \
    .with_capabilities([ModelCapability.EMBEDDING]) \
    .with_icon("/icons/ai-models/nvidia-color.svg") \
    .with_color("#76B900") \
    .add_field(API_KEY, ModelCapability.EMBEDDING) \
    .add_field(_NVIDIA_MODEL, ModelCapability.EMBEDDING) \
    .add_field(_NVIDIA_ENDPOINT, ModelCapability.EMBEDDING) \
    .add_field(EMBEDDING_COMMON_TAIL[0], ModelCapability.EMBEDDING) \
    .add_field(EMBEDDING_COMMON_TAIL[1], ModelCapability.EMBEDDING) \
    .add_field(EMBEDDING_COMMON_TAIL[2], ModelCapability.EMBEDDING) \
    .build_decorator()
class NvidiaProvider:
    pass
