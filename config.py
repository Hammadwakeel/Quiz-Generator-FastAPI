from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ───────────────────────────────────────────────────────────────────────────
    # Chat & RAG Configuration
    # ───────────────────────────────────────────────────────────────────────────
    groq_api_key: str
    vectorstore_base_path: str = "./vectorstores"

    # ───────────────────────────────────────────────────────────────────────────
    # Hugging Face Hub
    # ───────────────────────────────────────────────────────────────────────────
    huggingfacehub_api_token: str

    # ───────────────────────────────────────────────────────────────────────────
    # MongoDB Configuration (Local)
    # ───────────────────────────────────────────────────────────────────────────
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_chat_db: str = "QuizAI"
    mongo_chat_collection: str = "chat_histories"

    # ───────────────────────────────────────────────────────────────────────────
    # FastAPI Server Configuration
    # ───────────────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ───────────────────────────────────────────────────────────────────────────
    # App Metadata (unchanged)
    # ───────────────────────────────────────────────────────────────────────────
    app_name: str = "AI Quiz Generator"
    app_version: str = "1.0.0"
    app_description: str = (
        "An AI-powered quiz generator that creates quizzes from documents and "
        "allows users to chat with the AI about those quizzes."
    )

    # ───────────────────────────────────────────────────────────────────────────
    # Tell Pydantic to load from .env and ignore extras
    # ───────────────────────────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Single shared Settings instance
settings = Settings()
