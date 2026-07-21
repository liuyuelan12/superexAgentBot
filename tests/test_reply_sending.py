"""Tests for the send path: HTML parse mode, splitting, and plain-text fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from telegram.constants import ParseMode
from telegram.error import BadRequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.handlers import _reply_long, _send_formatted  # noqa: E402


def fake_message():
    msg = AsyncMock()
    msg.reply_text = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_answer_is_sent_as_html():
    msg = fake_message()
    await _send_formatted(msg, "**现货费率** 0.1%")
    msg.reply_text.assert_awaited_once()
    kwargs = msg.reply_text.await_args.kwargs
    assert msg.reply_text.await_args.args[0] == "<b>现货费率</b> 0.1%"
    assert kwargs["parse_mode"] == ParseMode.HTML
    assert kwargs["disable_web_page_preview"] is True


@pytest.mark.asyncio
async def test_falls_back_to_plain_text_when_telegram_rejects_html():
    """A formatting bug must never swallow the answer."""
    msg = fake_message()
    msg.reply_text.side_effect = [BadRequest("Can't parse entities"), None]
    await _send_formatted(msg, "**粗体** 内容")
    assert msg.reply_text.await_count == 2
    second = msg.reply_text.await_args_list[1]
    assert second.args[0] == "粗体 内容"
    assert "parse_mode" not in second.kwargs


@pytest.mark.asyncio
async def test_falls_back_to_raw_text_when_conversion_crashes():
    msg = fake_message()
    with patch("bot.handlers.md_to_telegram_html", side_effect=RuntimeError("boom")):
        await _send_formatted(msg, "原始内容")
    msg.reply_text.assert_awaited_once()
    assert msg.reply_text.await_args.args[0] == "原始内容"


@pytest.mark.asyncio
async def test_fallback_keeps_the_pagination_suffix():
    msg = fake_message()
    msg.reply_text.side_effect = [BadRequest("bad"), None]
    await _send_formatted(msg, "**x**", suffix="\n\n(1/2)")
    assert msg.reply_text.await_args_list[1].args[0].endswith("(1/2)")


@pytest.mark.asyncio
async def test_short_answer_gets_no_pagination_marker():
    msg = fake_message()
    await _reply_long(msg, "简短回答")
    msg.reply_text.assert_awaited_once()
    assert msg.reply_text.await_args.args[0] == "简短回答"


@pytest.mark.asyncio
async def test_long_answer_is_split_and_numbered():
    msg = fake_message()
    await _reply_long(msg, "段落内容。\n\n" * 900)
    assert msg.reply_text.await_count > 1
    total = msg.reply_text.await_count
    for i, call in enumerate(msg.reply_text.await_args_list, 1):
        text = call.args[0]
        assert text.endswith(f"({i}/{total})")
        assert call.kwargs["parse_mode"] == ParseMode.HTML


@pytest.mark.asyncio
async def test_empty_answer_sends_nothing():
    msg = fake_message()
    await _reply_long(msg, "   ")
    msg.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_refusal_template_survives_the_html_path():
    """The refusal contains @SuperEx_Zendesk_Bot — underscores must not be eaten."""
    msg = fake_message()
    from llm.prompts import refusal_for

    await _reply_long(msg, refusal_for("zh"))
    sent = msg.reply_text.await_args.args[0]
    assert "@SuperEx_Zendesk_Bot" in sent
    assert "<i>" not in sent
