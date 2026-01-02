from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")

    APP_NAME: str = "Hate Speech Detection Backend"
    VERSION: str = "1.0.0"

    # Comma-separated list of allowed origins
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Path to lexicon json file
    KEYWORDS_PATH: str = "data/hate_keywords.json"

    # YouTube Data API key (optional)
    YOUTUBE_API_KEY: str | None = None

    # Qwen rationale model configuration
    QWEN_BASE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    QWEN_ADAPTER_PATH: str = "checkpoints/qwen_datasetA_stage2_lora_adapters"
    QWEN_TOKENIZER_PATH: str = "checkpoints/qwen_datasetA_stage2_tokenizer"
    QWEN_MAX_LENGTH: int = 512
    QWEN_MAX_NEW_TOKENS: int = 50
    QWEN_USE_4BIT: bool = True
    QWEN_OFFLOAD_DIR: str = "checkpoints/offload"

settings = Settings()
