from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    upload_directory: str = "uploads"
    openai_api_key: str
    openai_models: List[str] = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()