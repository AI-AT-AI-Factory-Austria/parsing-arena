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

    # Page images shown in the UI. Each pipeline renders at its own DPI as well.
    render_dpi: int = 144

    # Open models served by an OpenAI-compatible endpoint (e.g. vLLM). When an
    # endpoint URL is empty the model runs locally with transformers instead.
    # Field names are the model id with non-alphanumerics replaced by "_".
    rakedoc_nano_endpoint_url: str = ""
    rakedoc_nano_served_model: str = "rakedoc-nano"
    aeddix_alpine_ocr_endpoint_url: str = ""
    aeddix_alpine_ocr_served_model: str = "aeddix-alpine-ocr"
    florin_parser_nano_endpoint_url: str = ""
    florin_parser_nano_served_model: str = "florin-parser-nano"
    nemotron_3_5_parse_endpoint_url: str = ""
    nemotron_3_5_parse_served_model: str = "nemotron-parse"
    lightonocr_3_0_8b_endpoint_url: str = ""
    lightonocr_3_0_8b_served_model: str = "loocr-grounding"
    mineru2_5_pro_2605_endpoint_url: str = ""
    mineru2_5_pro_2605_served_model: str = ""
    infinity_parser2_flash_endpoint_url: str = ""
    infinity_parser2_flash_served_model: str = "infly/Infinity-Parser2-Flash"
    ovisocr3_endpoint_url: str = ""
    ovisocr3_served_model: str = "ovisocr3"

    # Tesseract: path to the tesseract executable (empty = PATH or the default
    # Windows install folder) and its language codes, e.g. "deu+eng".
    tesseract_cmd: str = ""
    tesseract_lang: str = "eng"
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
    # Upper bound on checkpoints kept in memory at once; the least recently used
    # one is unloaded first. On machines whose GPU shares system RAM this keeps
    # a full arena run from exhausting memory.
    max_loaded_models: int = 2
    # Models known to crash AMD's ROCm stack are refused on ROCm unless this is set.
    allow_unstable_rocm_models: bool = False

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
