from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field("llama3.1", alias="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(
        "nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL"
    )
    vector_db_path: str = Field("vector_db/faiss_index", alias="VECTOR_DB_PATH")
    app_db_path: str = Field("data/app.db", alias="APP_DB_PATH")
    app_secret_key: str = Field("change-this-in-production", alias="APP_SECRET_KEY")
    admin_username: str = Field("admin", alias="ADMIN_USERNAME")
    admin_password: str = Field("admin123", alias="ADMIN_PASSWORD")
    data_dir: str = Field("data", alias="DATA_DIR")
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    top_k: int = Field(2, alias="TOP_K")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_db_path)

    @property
    def documents_dir(self) -> Path:
        return Path(self.data_dir)

    @property
    def db_path(self) -> Path:
        return Path(self.app_db_path)


settings = Settings()