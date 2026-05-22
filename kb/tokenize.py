"""Multilingual tokenizer for BM25.

Splits Latin / Cyrillic / Arabic / Hebrew words by whitespace + punctuation, and
breaks CJK into single characters so 中文 / 日本語 receive per-char weighting.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
    )


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        if any(_is_cjk(c) for c in word):
            for c in word:
                if _is_cjk(c):
                    tokens.append(c)
                elif c.isalnum():
                    tokens.append(c)
        else:
            if len(word) >= 2 or word.isdigit():
                tokens.append(word)
    return tokens
