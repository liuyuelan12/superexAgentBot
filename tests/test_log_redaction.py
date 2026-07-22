"""Tests for credential redaction in logs.

Production logs carried the live bot token on every httpx line. These tests pin
the masking so a refactor cannot quietly reopen that hole.
"""

from __future__ import annotations

import logging

import pytest

from bot import log_redaction
from bot.log_redaction import (
    RedactingFilter,
    install_log_redaction,
    redact,
    register_secret,
)


@pytest.fixture(autouse=True)
def _isolate_registered_secrets():
    """Keep registrations from leaking between tests."""
    saved = log_redaction._literal_secrets
    yield
    log_redaction._literal_secrets = saved

# Shaped like a real token but not one: 10-digit id, 35-char secret.
FAKE_TOKEN = "1234567890:AAFakeTokenValueForTestingOnly12345"


class TestRedact:
    def test_masks_telegram_token_in_url(self):
        line = f'HTTP Request: POST https://api.telegram.org/bot{FAKE_TOKEN}/getMe "200 OK"'
        out = redact(line)
        assert "AAFakeTokenValueForTestingOnly12345" not in out
        assert "REDACTED" in out

    def test_keeps_numeric_bot_id_for_correlation(self):
        out = redact(f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage")
        assert "/bot1234567890:" in out

    def test_masks_every_occurrence(self):
        line = f"first bot{FAKE_TOKEN}/getMe then /bot{FAKE_TOKEN}/sendMessage"
        assert redact(line).count("REDACTED") == 2

    def test_masks_provider_keys(self):
        assert "gsk_" not in redact("key=gsk_abcdefghij0123456789ABCDEFGHIJ")
        assert "sk-4" not in redact("key=sk-4990abcdefghij0123456789")

    def test_leaves_ordinary_text_alone(self):
        line = "Retriever ready: chroma=3376, bm25=3376"
        assert redact(line) == line

    def test_does_not_eat_normal_urls(self):
        line = "see https://www.superex.com/userCenter/fee/trading"
        assert redact(line) == line


class TestRegisteredSecrets:
    def test_registered_value_is_masked_without_url_context(self):
        register_secret("some-opaque-secret-value-1234")
        assert "some-opaque-secret-value-1234" not in redact(
            "config dump: token=some-opaque-secret-value-1234"
        )

    def test_short_values_are_ignored(self):
        # masking "prod" everywhere would shred logs and protect nothing
        register_secret("prod")
        assert redact("environment=prod") == "environment=prod"

    def test_none_is_ignored(self):
        register_secret(None)
        assert redact("nothing to do") == "nothing to do"


class TestFilter:
    def _record(self, msg, args=()):
        return logging.LogRecord("t", logging.INFO, __file__, 1, msg, args, None)

    def test_filter_masks_formatted_message(self):
        record = self._record("calling %s", (f"https://api.telegram.org/bot{FAKE_TOKEN}/getMe",))
        assert RedactingFilter().filter(record) is True
        assert "AAFakeTokenValueForTestingOnly12345" not in record.getMessage()

    def test_filter_always_keeps_the_record(self):
        # redaction must never silently drop a log line
        assert RedactingFilter().filter(self._record("plain")) is True

    def test_broken_format_string_is_not_dropped(self):
        record = self._record("needs %s %s", ("only-one",))
        assert RedactingFilter().filter(record) is True

    def test_install_attaches_to_handlers(self, caplog):
        logger = logging.getLogger("redaction-install-test")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        try:
            install_log_redaction(logger)
            assert any(isinstance(f, RedactingFilter) for f in handler.filters)
        finally:
            logger.removeHandler(handler)


class TestEndToEnd:
    def test_token_never_reaches_the_stream(self):
        import io

        stream = io.StringIO()
        logger = logging.getLogger("redaction-e2e-test")
        logger.propagate = False
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            install_log_redaction(logger)
            logger.info("POST https://api.telegram.org/bot%s/getMe", FAKE_TOKEN)
            handler.flush()
            assert "AAFakeTokenValueForTestingOnly12345" not in stream.getvalue()
        finally:
            logger.removeHandler(handler)
