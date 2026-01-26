"""LLM factory for multi-provider support."""

from langchain_core.language_models import BaseChatModel

from diabetic_api.core.config import Settings, LLMProvider, get_settings


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """
    Get configured LLM instance based on settings.
    
    Supports OpenAI and Google Gemini providers.
    
    Args:
        settings: Application settings (uses default if not provided)
        
    Returns:
        Configured chat model instance
        
    Raises:
        ValueError: If provider is not configured or unsupported
    """
    if settings is None:
        settings = get_settings()
    
    match settings.llm_provider:
        case LLMProvider.GEMINI:
            return _get_gemini(settings)
        case LLMProvider.OPENAI:
            return _get_openai(settings)
        case _:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def _get_openai(settings: Settings) -> BaseChatModel:
    """Get OpenAI chat model."""
    from langchain_openai import ChatOpenAI
    
    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key not configured. "
            "Set OPENAI_API_KEY in your .env file."
        )
    
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )


def _get_gemini(settings: Settings) -> BaseChatModel:
    """Get Google Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not settings.google_api_key:
        raise ValueError(
            "Google API key not configured. "
            "Set GOOGLE_API_KEY in your .env file."
        )
    
    # Use model name directly - langchain-google-genai handles the format
    # Confirmed working models:
    #   - gemini-2.5-flash (latest, best value)
    #   - gemini-2.0-flash (stable, fast)
    #   - gemini-1.5-flash (fast, cheap)
    #   - gemini-1.5-pro (better reasoning)
    model_name = settings.gemini_model
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
    )


def get_llm_info(settings: Settings | None = None) -> dict:
    """
    Get information about the configured LLM.
    
    Args:
        settings: Application settings
        
    Returns:
        Dict with provider info
    """
    if settings is None:
        settings = get_settings()
    
    return {
        "provider": settings.llm_provider.value,
        "model": (
            settings.gemini_model 
            if settings.llm_provider == LLMProvider.GEMINI 
            else settings.openai_model
        ),
        "configured": settings.is_llm_configured,
        "temperature": settings.llm_temperature,
    }

