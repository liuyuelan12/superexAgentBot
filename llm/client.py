"""LLM client with Groq primary and DeepSeek fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from groq import AsyncGroq
from openai import AsyncOpenAI

from config import ANSWER_MODEL, DEEPSEEK_MODEL, ROUTER_MODEL, load_secrets

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: Literal["groq", "deepseek"]
    fallback: bool
    model: str


class LLMClient:
    """Wrap Groq + DeepSeek with automatic fallback on error."""

    def __init__(self) -> None:
        secrets = load_secrets()
        self._groq = AsyncGroq(api_key=secrets.groq_api_key)
        self._deepseek: AsyncOpenAI | None = None
        if secrets.deepseek_api_key:
            self._deepseek = AsyncOpenAI(
                api_key=secrets.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        purpose: Literal["answer", "router"] = "answer",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResult:
        groq_model = ANSWER_MODEL if purpose == "answer" else ROUTER_MODEL
        payload: dict[str, Any] = {
            "model": groq_model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = await self._groq.chat.completions.create(**payload)
            return ChatResult(
                text=resp.choices[0].message.content or "",
                provider="groq",
                fallback=False,
                model=groq_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq call failed (%s); attempting DeepSeek fallback", exc)

        if self._deepseek is None:
            raise RuntimeError("Groq failed and DeepSeek fallback is not configured")

        ds_payload = dict(payload)
        ds_payload["model"] = DEEPSEEK_MODEL
        resp = await self._deepseek.chat.completions.create(**ds_payload)
        return ChatResult(
            text=resp.choices[0].message.content or "",
            provider="deepseek",
            fallback=True,
            model=DEEPSEEK_MODEL,
        )

    async def chat_json(
        self,
        messages: list[ChatMessage],
        *,
        purpose: Literal["answer", "router"] = "router",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> tuple[dict[str, Any], ChatResult]:
        result = await self.chat(
            messages,
            purpose=purpose,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            data = json.loads(result.text)
        except json.JSONDecodeError:
            data = _salvage_json(result.text)
        return data, result


def _salvage_json(text: str) -> dict[str, Any]:
    """Best-effort recovery when the model returns JSON with surrounding noise."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
