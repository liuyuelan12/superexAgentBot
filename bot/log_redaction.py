"""Strip credentials out of log records before they are written.

The Telegram Bot API carries its token in the URL *path*, so httpx's INFO line
for every single API call published the full token to the Railway log stream:

    HTTP Request: POST https://api.telegram.org/bot<id>:<token>/getMe "200 OK"

Anyone able to read the deploy logs could take over the bot. Lowering httpx to
WARNING would hide that one line, but the same token also rides along in
python-telegram-bot exception text and in tracebacks, so the fix belongs at the
logging layer where it catches every path at once.

Filters are installed on the root *handlers* rather than on a logger: a
Logger's filters only see records that logger created, while a Handler's filters
see everything that reaches it, including records propagated up from
third-party libraries we do not control.
"""

from __future__ import annotations

import logging
import re

_MASK = "***REDACTED***"

# bot8627912069:AA...  — keep the numeric bot id, drop the secret. The id alone is
# public (it is the bot's user id) and keeps log lines correlatable. The leading
# slash is optional because the token also appears without one in library error
# text, and a redactor should fail closed.
_TELEGRAM_TOKEN_RE = re.compile(r"\b(bot\d{5,}):[A-Za-z0-9_\-]{20,}")

# Provider keys, in case one is ever interpolated into a message or traceback.
_PROVIDER_KEY_RE = re.compile(r"\b(?:gsk_|sk-|hf_)[A-Za-z0-9_\-]{16,}")

# Literal secret values registered at startup. Patterns cannot recognise a bare
# token printed with no surrounding context (a config dump, say), so the real
# values are masked by exact match as a backstop.
_literal_secrets: frozenset[str] = frozenset()


def register_secret(value: str | None) -> None:
    """Mask every future occurrence of ``value`` verbatim.

    Short values are ignored: masking a 4-character string would shred ordinary
    log text without protecting anything worth protecting.
    """
    global _literal_secrets
    if value and len(value) >= 12:
        _literal_secrets = _literal_secrets | {value}


def redact(text: str) -> str:
    """Mask any credential recognised in ``text``."""
    for secret in _literal_secrets:
        if secret in text:
            text = text.replace(secret, _MASK)
    text = _TELEGRAM_TOKEN_RE.sub(rf"\1:{_MASK}", text)
    return _PROVIDER_KEY_RE.sub(_MASK, text)


class RedactingFilter(logging.Filter):
    """Rewrite each record's message with credentials masked.

    ``record.getMessage()`` is resolved here and ``args`` cleared, which makes
    formatting eager. That trades a little work per record for the guarantee
    that no later formatting step can reintroduce a raw secret — acceptable at
    this bot's log volume.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken format string must not drop the log
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def install_log_redaction(logger: logging.Logger | None = None) -> RedactingFilter:
    """Attach the filter to every handler of ``logger`` (root by default)."""
    target = logger if logger is not None else logging.getLogger()
    log_filter = RedactingFilter()
    for handler in target.handlers:
        handler.addFilter(log_filter)
    return log_filter
