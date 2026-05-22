"""Reply-chain history walker for multi-turn context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from telegram import Message

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    role: Role
    text: str


def build_history(message: Message, bot_id: int, limit: int = 5) -> list[Turn]:
    """Telegram only exposes the immediate reply_to_message; this returns 0–1 prior turns."""
    history: list[Turn] = []
    parent = message.reply_to_message
    if parent is not None and parent.text:
        is_bot = bool(parent.from_user and parent.from_user.id == bot_id)
        history.append(Turn(role="assistant" if is_bot else "user", text=parent.text))
    return history[-limit:]


def format_history(history: list[Turn]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"[{t.role}] {t.text}" for t in history)
