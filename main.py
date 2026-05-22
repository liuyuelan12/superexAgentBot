"""SuperEx Agent Bot — entry point."""

from __future__ import annotations

import logging
import sys

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from bot.handlers import AskHandler, on_help, on_start
from config import load_runtime_config, load_secrets
from kb.retriever import Retriever
from llm.client import LLMClient

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("superex.agent.main")


def build_app() -> Application:
    secrets = load_secrets()
    runtime = load_runtime_config()

    try:
        retriever = Retriever()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(2)

    llm = LLMClient()
    ask = AskHandler(llm=llm, retriever=retriever, runtime=runtime)

    app = Application.builder().token(secrets.bot_token).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("ask", ask.on_ask))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, ask.on_message)
    )
    return app


def main() -> None:
    app = build_app()
    logger.info("SuperExAgentBot starting (polling)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
