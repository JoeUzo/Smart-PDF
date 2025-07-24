from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    upload_directory: str = "uploads"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
