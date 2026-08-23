"""
LLM provider factory.

Reads ``LLM_PROVIDER`` from the environment (default: ``azure``) and returns
the corresponding LangChain chat-model instance.  All providers share one
common contract — LangChain's ``BaseChatModel`` — so the rest of the codebase
(``bind_tools``, ``with_structured_output``, etc.) works unchanged.

Supported providers
-------------------
- **azure** — ``AzureChatOpenAI`` (production default)
- **gemini** — ``ChatGoogleGenerativeAI`` (free-tier / evaluation)
- **groq** — ``ChatGroq`` (fast inference / evaluation)
"""
import logging
import os

logger = logging.getLogger(__name__)

# Lazy imports — only the selected provider's package is imported at runtime,
# so users aren't forced to install all three SDK packages.


def get_llm():
    """
    Build and return a LangChain chat model based on ``LLM_PROVIDER`` env var.

    Environment variables per provider
    -----------------------------------
    **azure** (default):
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION, AZURE_OPENAI_TEMPERATURE

    **gemini**:
        GOOGLE_API_KEY, GEMINI_MODEL (default: gemini-2.5-flash),
        GEMINI_TEMPERATURE (default: 0)

    **groq**:
        GROQ_API_KEY, GROQ_MODEL (default: llama-3.3-70b-versatile),
        GROQ_TEMPERATURE (default: 0)
    """
    provider = os.getenv("LLM_PROVIDER", "azure").lower().strip()

    if provider == "azure":
        return _build_azure()
    elif provider == "gemini":
        return _build_gemini()
    elif provider == "groq":
        return _build_groq()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            f"Supported: azure, gemini, groq"
        )


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

def _build_azure():
    from langchain_openai import AzureChatOpenAI
    from sql_agent.config.settings import (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_TEMPERATURE,
    )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=AZURE_OPENAI_TEMPERATURE,
    )
    logger.info(
        "[llm_factory] Provider=azure | deployment=%s | temperature=%s",
        AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_TEMPERATURE,
    )
    return llm


# ── Google Gemini ─────────────────────────────────────────────────────────────

def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key     = os.getenv("GOOGLE_API_KEY", "")
    model       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0"))

    if not api_key:
        raise ValueError(
            "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set. "
            "Get one at https://aistudio.google.com/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )
    logger.info(
        "[llm_factory] Provider=gemini | model=%s | temperature=%s",
        model,
        temperature,
    )
    return llm


# ── Groq ──────────────────────────────────────────────────────────────────────

def _build_groq():
    from langchain_groq import ChatGroq

    api_key     = os.getenv("GROQ_API_KEY", "")
    model       = os.getenv("GROQ_MODEL") or os.getenv("GROQ_MODEL_NAME") or "openai/gpt-oss-120b"
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0"))

    if not api_key:
        raise ValueError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get one at https://console.groq.com/keys"
        )

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
    logger.info(
        "[llm_factory] Provider=groq | model=%s | temperature=%s",
        model,
        temperature,
    )
    return llm
