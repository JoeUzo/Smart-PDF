from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    upload_directory: str = "uploads"
    openai_api_key: str
    openai_models: List[str] = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    upload_dir_ttl_hours: int = 24
    redis_url: str = "redis://localhost:6379/0"
    log_dir: str = "logs"
    email: str = None
    email_key: str = None

    # OCR Configuration
    ocr_enabled: bool = True
    ocr_dpi: int = 300
    ocr_lang: str = "eng"
    ocr_config: str = r'--oem 3 --psm 6'
    ocr_save_debug_images: bool = True
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()