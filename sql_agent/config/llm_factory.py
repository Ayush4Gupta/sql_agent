"""
LLM provider factory with round-robin multi-provider support.

Reads ``LLM_PROVIDER`` from the environment.  If a single provider name is
given (e.g. ``groq``), it returns that provider's model directly — identical
to the original behaviour.

If a comma-separated list is given (e.g. ``groq,gemini``), it returns a
``RoundRobinLLM`` wrapper that cycles across providers on each invocation
and automatically falls back to the next provider on errors (rate-limit,
network, etc.).

Circuit breaker: when a provider returns a daily rate-limit error (TPD)
multiple times consecutively, it's marked "dead" and skipped for a
cooldown period.  This prevents wasting time cycling through exhausted
Groq keys when Gemini (separate quota pool) is still available.

Supported providers
-------------------
- **azure** — ``AzureChatOpenAI`` (production default)
- **gemini** — ``ChatGoogleGenerativeAI`` (free-tier / evaluation)
- **groq** — ``ChatGroq`` (fast inference / evaluation)
"""
import logging
import os
import time
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Round-Robin LLM Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RoundRobinLLM(BaseChatModel):
    """Cycles through a list of LLMs, falling back on errors.

    On each ``invoke`` / ``_generate`` call, the wrapper picks the next LLM
    in the rotation.  If that LLM raises an exception (rate-limit, network,
    validation error, etc.), it tries the remaining LLMs in order before
    giving up.  This is intentionally broad — any transient error on one
    provider is reason enough to try another.

    ``bind_tools`` and ``with_structured_output`` return new ``RoundRobinLLM``
    instances wrapping the bound/structured variants of all inner LLMs, so
    tool-calling and structured-output pipelines get the same rotation.
    """

    models: List[Any]  # BaseChatModel or Runnable (after bind_tools)
    provider_names: List[str]

    class Config:
        arbitrary_types_allowed = True

    # Circuit-breaker constants
    CB_FAIL_THRESHOLD: int = 2     # consecutive rate-limit failures to trip
    CB_COOLDOWN_SECS: float = 300  # 5 min cooldown for transient limits

    def model_post_init(self, __context: Any) -> None:
        """Initialise the rotation counter and circuit-breaker state."""
        super().model_post_init(__context)
        # Plain instance attrs — not Pydantic fields, so no deepcopy issues
        object.__setattr__(self, "_rr_counter", 0)
        # Circuit breaker: {index: {"consecutive_fails": int, "dead_until": float}}
        object.__setattr__(self, "_cb_state", {})

    @property
    def _llm_type(self) -> str:
        return "round_robin"

    def _next_index(self) -> int:
        idx = self._rr_counter % len(self.models)
        object.__setattr__(self, "_rr_counter", self._rr_counter + 1)
        return idx

    def _is_dead(self, idx: int) -> bool:
        """Check if a provider is circuit-broken (too many rate-limit failures)."""
        state = self._cb_state.get(idx)
        if not state:
            return False
        if state["consecutive_fails"] >= self.CB_FAIL_THRESHOLD:
            if time.time() < state["dead_until"]:
                return True
            # Cooldown expired — revive for a retry
            self._cb_state.pop(idx, None)
            logger.info(
                "[round_robin] Reviving provider '%s' after cooldown",
                self.provider_names[idx],
            )
        return False

    def _mark_rate_limited(self, idx: int, error: Exception) -> None:
        """Record a rate-limit failure; trip circuit if threshold reached."""
        state = self._cb_state.get(idx, {"consecutive_fails": 0, "dead_until": 0})
        state["consecutive_fails"] += 1

        # Check if this is a daily (TPD) limit vs transient per-minute limit
        err_str = str(error)
        is_daily = "tokens per day" in err_str.lower() or "tpd" in err_str.lower()

        if state["consecutive_fails"] >= self.CB_FAIL_THRESHOLD:
            # Daily limits: circuit-break for 30 minutes (no point retrying sooner)
            # Transient limits: shorter cooldown
            cooldown = 1800 if is_daily else self.CB_COOLDOWN_SECS
            state["dead_until"] = time.time() + cooldown
            logger.warning(
                "[round_robin] Circuit OPEN for '%s' — %d consecutive rate-limit "
                "failures, dead for %ds%s",
                self.provider_names[idx],
                state["consecutive_fails"],
                cooldown,
                " (daily TPD exhausted)" if is_daily else "",
            )

        self._cb_state[idx] = state

    def _mark_success(self, idx: int) -> None:
        """Clear circuit-breaker state on success."""
        self._cb_state.pop(idx, None)

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Check if an error is a rate-limit (429) error."""
        err_str = str(error)
        return "429" in err_str or "rate_limit" in err_str.lower()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Required by BaseChatModel but we override invoke() instead.
        # This is only called if someone bypasses invoke() directly.
        from langchain_core.outputs import ChatGeneration
        result = self.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=result)])

    def invoke(self, input, config=None, **kwargs):
        """Round-robin invoke with circuit breaker.

        Skips providers that are circuit-broken (dead from repeated rate-limit
        failures), tries remaining live providers in rotation order.
        """
        start_idx = self._next_index()
        last_error = None
        tried = 0
        skipped_dead = 0

        for offset in range(len(self.models)):
            idx = (start_idx + offset) % len(self.models)
            provider = self.provider_names[idx]

            # Circuit breaker: skip dead providers
            if self._is_dead(idx):
                skipped_dead += 1
                continue

            llm = self.models[idx]
            tried += 1
            try:
                result = llm.invoke(input, config=config, **kwargs)
                self._mark_success(idx)
                if tried > 1:
                    logger.info(
                        "[round_robin] Succeeded on fallback provider '%s'",
                        provider,
                    )
                return result
            except Exception as e:
                last_error = e
                if self._is_rate_limit_error(e):
                    self._mark_rate_limited(idx, e)
                logger.warning(
                    "[round_robin] Provider '%s' failed: %s — trying next",
                    provider,
                    str(e)[:200],
                )
                continue

        # All providers either dead or failed
        live_count = len(self.models) - skipped_dead
        logger.error(
            "[round_robin] All providers exhausted. %d tried, %d circuit-broken. "
            "Last error: %s",
            tried, skipped_dead, last_error,
        )
        raise last_error

    def bind_tools(self, tools, **kwargs) -> "RoundRobinLLM":
        """Bind tools to ALL inner LLMs and return a new RoundRobinLLM."""
        bound = [m.bind_tools(tools, **kwargs) for m in self.models]
        return RoundRobinLLM(
            models=bound,
            provider_names=self.provider_names,
        )

    def with_structured_output(self, schema, **kwargs) -> "RoundRobinLLM":
        """Apply structured output to ALL inner LLMs and return a new RoundRobinLLM."""
        structured = [m.with_structured_output(schema, **kwargs) for m in self.models]
        return RoundRobinLLM(
            models=structured,
            provider_names=self.provider_names,
        )

    @property
    def _identifying_params(self) -> dict:
        return {
            "providers": self.provider_names,
            "model_count": len(self.models),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_llm():
    """
    Build and return a LangChain chat model based on ``LLM_PROVIDER`` env var.

    Single provider
    ~~~~~~~~~~~~~~~
    ``LLM_PROVIDER=groq`` → returns a ``ChatGroq`` instance directly.

    Multi-provider (round-robin)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``LLM_PROVIDER=groq,gemini`` → returns a ``RoundRobinLLM`` that cycles
    between Groq and Gemini on each call, falling back automatically.

    Environment variables per provider
    -----------------------------------
    **azure** (default):
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION, AZURE_OPENAI_TEMPERATURE

    **gemini**:
        GOOGLE_API_KEY, GEMINI_MODEL (default: gemini-3.5-flash-lite),
        GEMINI_TEMPERATURE (default: 0)

    **groq**:
        GROQ_API_KEY, GROQ_MODEL (default: openai/gpt-oss-120b),
        GROQ_TEMPERATURE (default: 0)
    """
    raw = os.getenv("LLM_PROVIDER", "azure").strip()
    providers = [p.strip().lower() for p in raw.split(",") if p.strip()]

    if not providers:
        raise ValueError("LLM_PROVIDER is empty.")

    # Collect all (name, llm) pairs — a single provider may expand to
    # multiple instances (e.g. groq with GROQ_API_KEYS).
    models = []
    names = []
    for p in providers:
        try:
            instances = _build_provider_instances(p)
            for name, llm in instances:
                models.append(llm)
                names.append(name)
        except Exception as e:
            logger.warning(
                "[llm_factory] Could not build provider '%s': %s — skipping",
                p, e,
            )

    if not models:
        raise ValueError(
            f"LLM_PROVIDER={raw!r} but no providers could be initialised."
        )

    # Single instance — return directly, no wrapper overhead
    if len(models) == 1:
        return models[0]

    wrapper = RoundRobinLLM(models=models, provider_names=names)
    logger.info(
        "[llm_factory] Round-robin mode | %d instances: %s",
        len(models),
        names,
    )
    return wrapper


_SUPPORTED_PROVIDERS = {"azure", "gemini", "groq"}


def _build_provider_instances(provider: str) -> list[tuple[str, BaseChatModel]]:
    """Build LLM instance(s) for a provider.

    Returns a list of (display_name, llm) tuples.  Most providers return
    exactly one; groq may return many if GROQ_API_KEYS has multiple keys.
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )

    if provider == "azure":
        return [("azure", _build_azure())]
    elif provider == "gemini":
        return _build_gemini_instances()
    elif provider == "groq":
        return _build_groq_instances()


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

def _build_gemini_instances() -> list[tuple[str, BaseChatModel]]:
    """Build one ChatGoogleGenerativeAI per API key.

    Reads keys from GOOGLE_API_KEYS (comma-separated) first, falling back
    to GOOGLE_API_KEY (single key or comma-separated). Each key creates a
    separate ChatGoogleGenerativeAI instance for the round-robin pool.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    model       = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0"))

    # Collect keys: prefer GOOGLE_API_KEYS, fall back to GOOGLE_API_KEY
    raw_keys = os.getenv("GOOGLE_API_KEYS", "") or os.getenv("GOOGLE_API_KEY", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not keys:
        raise ValueError(
            "LLM_PROVIDER=gemini but neither GOOGLE_API_KEYS nor GOOGLE_API_KEY is set. "
            "Get one at https://aistudio.google.com/apikey"
        )

    instances = []
    for i, key in enumerate(keys):
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temperature,
        )
        name = f"gemini-{i+1}" if len(keys) > 1 else "gemini"
        instances.append((name, llm))

    logger.info(
        "[llm_factory] Provider=gemini | model=%s | temperature=%s | %d key(s)",
        model,
        temperature,
        len(keys),
    )
    return instances


# ── Groq ──────────────────────────────────────────────────────────────────────

def _build_groq_instances() -> list[tuple[str, BaseChatModel]]:
    """Build one ChatGroq per API key.

    Reads keys from GROQ_API_KEYS (comma-separated) first, falling back
    to the legacy GROQ_API_KEY (single key).  Each key creates a separate
    ChatGroq instance for the round-robin pool.
    """
    from langchain_groq import ChatGroq

    model       = os.getenv("GROQ_MODEL") or os.getenv("GROQ_MODEL_NAME") or "openai/gpt-oss-120b"
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0"))

    # Collect keys: prefer GROQ_API_KEYS, fall back to GROQ_API_KEY
    raw_keys = os.getenv("GROQ_API_KEYS", "") or os.getenv("GROQ_API_KEY", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not keys:
        raise ValueError(
            "LLM_PROVIDER=groq but neither GROQ_API_KEYS nor GROQ_API_KEY is set. "
            "Get keys at https://console.groq.com/keys"
        )

    instances = []
    for i, key in enumerate(keys):
        llm = ChatGroq(
            model=model,
            api_key=key,
            temperature=temperature,
        )
        name = f"groq-{i+1}" if len(keys) > 1 else "groq"
        instances.append((name, llm))

    logger.info(
        "[llm_factory] Provider=groq | model=%s | temperature=%s | %d key(s)",
        model,
        temperature,
        len(keys),
    )
    return instances
