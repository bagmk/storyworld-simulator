"""
LLM Client abstraction for the AI Story Simulation Engine.

Supports:
  - OpenAI GPT-4, GPT-4o, GPT-4o-mini (and any compatible endpoint)
  - Budget tracking per episode
  - Automatic model tiering: cheap model for routine interactions,
    premium model for Director AI and critical moments

Usage:
    from llm_client import LLMClient
    client = LLMClient(model="gpt-4o-mini", budget_usd=2.00)
    response = client.chat(messages=[...])
"""

from __future__ import annotations
import os
import time
import logging
import json
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Model pricing (USD per 1K tokens, input/output) ──────────────────────────
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":             (0.005,  0.015),
    "gpt-4o-mini":        (0.00015, 0.0006),
    "gpt-4-turbo":        (0.01,   0.03),
    "gpt-4":              (0.03,   0.06),
    "gpt-3.5-turbo":      (0.0005, 0.0015),
    # Add more models here as needed
}

DEFAULT_CHEAP_MODEL   = "gpt-4o-mini"
DEFAULT_PREMIUM_MODEL = "gpt-4o"
# GPT-5 Responses calls can consume output budget before emitting visible text.
# Start with a safer floor to avoid frequent empty-first retries.
MIN_GPT5_OUTPUT_TOKENS = 800


@dataclass
class UsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    purpose: str       # e.g., "agent_turn", "director_check", "narrative_gen"


@dataclass
class LLMClient:
    """
    Wraps the OpenAI API with budget tracking and model tiering.

    Parameters
    ----------
    model : str
        Default model for all calls.
    premium_model : str
        Model used when use_premium=True (Director AI, key moments).
    budget_usd : float
        Hard stop when cumulative cost exceeds this.
    api_key : str | None
        Falls back to OPENAI_API_KEY env var.
    base_url : str | None
        Override for compatible endpoints (e.g., Azure, local).
    max_retries : int
        Number of retry attempts on transient errors.
    """

    model: str = DEFAULT_CHEAP_MODEL
    premium_model: str = DEFAULT_PREMIUM_MODEL
    budget_usd: float = 5.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    temperature: float = 0.8

    _spent_usd: float = field(default=0.0, init=False)
    _usage_log: list[UsageRecord] = field(default_factory=list, init=False)
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logger.warning(
                "No OPENAI_API_KEY found. Set the env var or pass api_key=... "
                "to LLMClient. Calls will fail until a key is provided."
            )
        try:
            from openai import OpenAI
            kwargs = {"api_key": key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget_usd - self._spent_usd)

    @property
    def usage_log(self) -> list[UsageRecord]:
        return list(self._usage_log)

    def budget_summary(self) -> dict:
        return {
            "budget_usd": self.budget_usd,
            "spent_usd": round(self._spent_usd, 6),
            "remaining_usd": round(self.remaining_budget, 6),
            "call_count": len(self._usage_log),
            "breakdown": [
                {"purpose": r.purpose, "model": r.model, "cost": round(r.cost_usd, 6)}
                for r in self._usage_log
            ],
        }

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        use_premium: bool = False,
        purpose: str = "general",
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
    ) -> str:
        """
        Send a chat completion request.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...}
        system   : optional system prompt prepended automatically
        use_premium : if True, use premium_model instead of model
        purpose  : label for budget logging
        temperature : overrides instance default if provided
        max_tokens : max response tokens

        Returns
        -------
        str : the assistant's reply text
        """
        if self._spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"Budget of ${self.budget_usd:.2f} exceeded "
                f"(spent ${self._spent_usd:.4f})."
            )

        model = self.premium_model if use_premium else self.model
        temp  = temperature if temperature is not None else self.temperature

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        last_error: Exception | None = None
        token_param = self._token_param_name(model)
        response_max_tokens = max_tokens
        if model.startswith("gpt-5"):
            response_max_tokens = max(response_max_tokens, MIN_GPT5_OUTPUT_TOKENS)
        for attempt in range(self.max_retries):
            try:
                if model.startswith("gpt-5"):
                    payload = {
                        "model": model,
                        "input": self._to_responses_input(full_messages),
                        "max_output_tokens": response_max_tokens,
                    }
                    reasoning_effort = self._default_reasoning_effort(model)
                    if reasoning_effort:
                        payload["reasoning"] = {"effort": reasoning_effort}

                    resp = self._client.responses.create(**payload)
                    status, incomplete_reason, error_text = \
                        self._extract_responses_status(resp)
                    if status == "failed":
                        raise RuntimeError(
                            f"Responses API call failed for {model}: "
                            f"{error_text or 'unknown error'}"
                        )

                    text = self._extract_text_from_responses(resp)
                    self._record_usage(resp, model, purpose)

                    if (
                        status == "incomplete"
                        and incomplete_reason == "max_output_tokens"
                        and not text.strip()
                        and attempt < self.max_retries - 1
                    ):
                        next_limit = min(response_max_tokens * 2, 16000)
                        if next_limit > response_max_tokens:
                            logger.warning(
                                "Model %s ran out of output tokens before producing visible text; "
                                "retrying with max_output_tokens=%d (was %d).",
                                model, next_limit, response_max_tokens,
                            )
                            response_max_tokens = next_limit
                            continue

                    if status == "incomplete":
                        logger.warning(
                            "Model %s returned incomplete response (reason=%s).",
                            model, incomplete_reason or "unknown",
                        )

                    if not text.strip():
                        logger.warning(
                            "Model %s returned an empty text response (status=%s, reason=%s).",
                            model, status or "unknown", incomplete_reason or "n/a",
                        )
                    return text

                payload = {
                    "model": model,
                    "messages": full_messages,
                    token_param: max_tokens,
                }
                if self._supports_custom_temperature(model):
                    payload["temperature"] = temp
                resp = self._client.chat.completions.create(
                    **payload
                )
                text = self._extract_text(resp)
                self._record_usage(resp, model, purpose)
                return text
            except Exception as exc:
                last_error = exc
                err = str(exc)
                # Some models (e.g., GPT-5) reject max_tokens and require
                # max_completion_tokens on chat.completions.
                if "Unsupported parameter: 'max_tokens'" in err and token_param == "max_tokens":
                    token_param = "max_completion_tokens"
                    logger.warning(
                        "Model %s requires max_completion_tokens; retrying with compatible token parameter.",
                        model,
                    )
                    continue
                if (
                    "Unsupported value: 'temperature'" in err
                    or "Only the default (1) value is supported" in err
                ):
                    logger.warning(
                        "Model %s does not support custom temperature; retrying with default temperature behavior.",
                        model,
                    )
                    temp = 1.0
                    continue
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise LLMCallError(
            f"LLM call failed after {self.max_retries} attempts."
        ) from last_error

    @staticmethod
    def _token_param_name(model: str) -> str:
        """
        Return the token-limit parameter accepted by the target model.
        GPT-5 family expects max_completion_tokens on chat.completions.
        """
        return "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"

    @staticmethod
    def _supports_custom_temperature(model: str) -> bool:
        """
        GPT-5 chat.completions currently supports only default temperature behavior.
        """
        return not model.startswith("gpt-5")

    @staticmethod
    def _default_reasoning_effort(model: str) -> Optional[str]:
        """
        Reduce token burn on older GPT-5 models that default to medium reasoning.
        GPT-5.1+ defaults are already tuned for lower overhead.
        """
        if model.startswith("gpt-5.1") or model.startswith("gpt-5.2"):
            return None
        if model.startswith("gpt-5"):
            return "low"
        return None

    @staticmethod
    def _to_responses_input(messages: list[dict]) -> list[dict]:
        """
        Convert chat-style messages to Responses API input.
        """
        out: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            content_type = "output_text" if role == "assistant" else "input_text"
            out.append({
                "role": role,
                "content": [{"type": content_type, "text": content}],
            })
        return out

    @staticmethod
    def _to_responses_messages(messages: list[dict]) -> list[dict]:
        """
        Convert chat-style messages into Responses API message objects.
        Use plain string content per message for widest compatibility.
        """
        out: list[dict] = []
        for m in messages:
            role = str(m.get("role", "user")).lower()
            if role not in ("system", "user", "assistant", "developer"):
                role = "user"
            content = str(m.get("content", ""))
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _extract_responses_status(response) -> tuple[str | None, str | None, str | None]:
        status = getattr(response, "status", None)

        incomplete = getattr(response, "incomplete_details", None)
        reason = None
        if isinstance(incomplete, dict):
            reason = incomplete.get("reason")
        else:
            reason = getattr(incomplete, "reason", None)

        error = getattr(response, "error", None)
        error_text = None
        if isinstance(error, dict):
            error_text = error.get("message") or error.get("code")
        else:
            error_text = getattr(error, "message", None) or getattr(error, "code", None)

        return status, reason, error_text

    @staticmethod
    def _extract_text(response) -> str:
        """
        Extract assistant text robustly from chat completion responses.
        Some model/SDK combos may return segmented content blocks.
        """
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                text_val = getattr(block, "text", None)
                if isinstance(text_val, str):
                    parts.append(text_val)
                    continue
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block.get("content"), str):
                        parts.append(block["content"])
            return "\n".join(p for p in parts if p).strip()

        # Final fallback: stringify content if present
        if content is None:
            return ""
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    @staticmethod
    def _extract_text_from_responses(response) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        output = getattr(response, "output", None)
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        if getattr(c, "type", "") == "output_text":
                            val = getattr(c, "text", "")
                            if val:
                                parts.append(val)
                        elif isinstance(c, dict) and c.get("type") == "output_text":
                            val = c.get("text", "")
                            if val:
                                parts.append(val)
            if parts:
                return "\n".join(parts).strip()

        # Generic fallback for SDK/schema differences
        dump = None
        try:
            if hasattr(response, "model_dump"):
                dump = response.model_dump()
            elif hasattr(response, "to_dict"):
                dump = response.to_dict()
        except Exception:
            dump = None

        if isinstance(dump, dict):
            extracted = LLMClient._extract_text_recursive(dump)
            if extracted:
                return extracted
        return ""

    @staticmethod
    def _extract_text_recursive(node) -> str:
        parts: list[str] = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("text", "output_text", "refusal") and isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                    walk(v)
            elif isinstance(x, list):
                for item in x:
                    walk(item)

        walk(node)
        # Preserve order while deduplicating
        seen: set[str] = set()
        unique: list[str] = []
        for p in parts:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        return "\n".join(unique).strip()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _record_usage(self, response, model: str, purpose: str) -> None:
        usage = response.usage
        prompt_tokens = (
            getattr(usage, "prompt_tokens", None)
            if usage is not None else None
        )
        completion_tokens = (
            getattr(usage, "completion_tokens", None)
            if usage is not None else None
        )
        # Responses API shape
        if prompt_tokens is None:
            prompt_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        if completion_tokens is None:
            completion_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0

        price_in, price_out = MODEL_PRICING.get(model, (0.0, 0.0))
        cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1000.0

        self._spent_usd += cost
        record = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            purpose=purpose,
        )
        self._usage_log.append(record)
        logger.debug(
            "LLM [%s|%s] prompt=%d completion=%d cost=$%.5f total=$%.4f",
            model, purpose, prompt_tokens, completion_tokens, cost, self._spent_usd,
        )


class BudgetExceededError(RuntimeError):
    pass


class LLMCallError(RuntimeError):
    pass
