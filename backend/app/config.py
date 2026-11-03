from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """All configuration comes from environment variables or the repo-level `.env`."""

    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LlamaParse (reference parser)
    llama_cloud_api_key: str = ""
    llama_cloud_base_url: str = "https://api.cloud.llamaindex.ai"
    llamaparse_tier: str = "agentic_plus"
    llamaparse_poll_interval_s: float = 2.0
    llamaparse_timeout_s: float = 1800.0

    # Storage
    data_dir: Path = REPO_DIR / "data"
    max_upload_mb: int = 100
    max_pages_per_experiment: int = 50

    # Page rendering (ParseBench renders PDFs at 144 DPI for these pipelines)
    render_dpi: int = 144

    # Open models served by an OpenAI-compatible endpoint (e.g. vLLM). When an
    # endpoint URL is empty the model runs locally with transformers instead.
    rakedoc_nano_endpoint_url: str = ""
    rakedoc_nano_served_model: str = "rakedoc-nano"
    aeddix_alpine_ocr_endpoint_url: str = ""
    aeddix_alpine_ocr_served_model: str = "aeddix-alpine-ocr"
    vllm_api_key: str = ""
    endpoint_timeout_s: float = 900.0
    kdl_max_concurrent: int = 8

    # Local inference (transformers)
    local_inference: bool = True
    local_device: str = "auto"  # auto | cuda | cpu
    local_dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    local_max_batch: int = 8
    local_max_model_len: int = 8192
    keep_models_loaded: bool = True

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    @property
    def database_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'arena.db').as_posix()}"

    @property
    def llamaparse_configured(self) -> bool:
        return bool(self.llama_cloud_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
