from langchain_core.language_model import BaseChatModel
from langchain_ollama import ChatOllama

from app.config import settings


def get_chat_llm() -> BaseChatModel:
    """Factory function to get the appropriate LLM based on configuration."""
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0,
        )

    elif provider == "claude" or provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment variables")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_chat_model,
            temperature=0,
        )

    elif provider == "huggingface":
        if not settings.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY is not set in environment variables")
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=settings.huggingface_chat_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            temperature=0,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Supported providers: ollama, openai, claude, huggingface"
        )
