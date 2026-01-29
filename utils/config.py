from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    WEBHOOK_DEVLOGS_CHANNEL: str = Field(default="", alias="WEBHOOK_DEVLOGS_CHANNEL")
    GITHUB_SECRET: bytes = Field(default=b"", alias="GITHUB_SECRET")
    PATH_BACKEND_HYDROPONIC: str = Field(default="", alias="PATH_BACKEND_HYDROPONIC")


config = Config()
