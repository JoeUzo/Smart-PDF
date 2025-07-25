from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    upload_directory: str = "uploads"
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()