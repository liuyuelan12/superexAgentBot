# SuperEx Agent Bot

Telegram AI customer-service bot for SuperEx. Backed by a hybrid RAG index
over `raw/客服`, `raw/官方教程`, and `wiki/`.

## Quickstart

```bash
cd Bot/SuperExAgentBot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Confirm the repo-root .env has BOT_TOKEN, GROQ_API_KEY, DEEPSEEK_API_KEY
# 2) Build the index (downloads bge-m3 ~2.3GB on first run)
python -m scripts.rebuild_index --force

# 3) Start the bot
python main.py
```

## Architecture

| Component | File |
|---|---|
| Telegram entry | `main.py` |
| Config / secrets | `config.py` |
| LLM client (Groq → DeepSeek fallback) | `llm/client.py` |
| Prompts + 9-language refusal | `llm/prompts.py` |
| Document loaders (CSV / FAQ / MD / HTML / PDF / wiki) | `kb/loader.py` |
| Hybrid indexer (chroma + BM25) | `kb/indexer.py` |
| Hybrid retriever | `kb/retriever.py` |
| Handlers (`/ask`, `/start`, `/help`, on_message) | `bot/handlers.py` |
| Router (intent + lang + query rewrite) | `bot/router.py` |
| Reply-chain history | `bot/context.py` |
| CLI for index rebuild | `scripts/rebuild_index.py` |

## How it answers

1. Strong triggers (`/ask`, @mention, reply-to-bot, private chat) go straight to retrieval.
2. Weak trigger (`ENABLE_WEAK_TRIGGER=True`, P1) runs router LLM on group messages.
3. Router returns `{is_question, lang, rewritten_query}`.
4. Retriever does hybrid vector (bge-m3) + BM25 fusion; `top_vec < 0.45` ⇒ refuse.
5. Answer LLM (`llama-3.3-70b-versatile`) generates with strict source-citation prompt.
6. Long replies are auto-chunked to ≤4000 chars; every Q&A is logged to `data/qa_log.jsonl`.

## Index sources

Set via `config.KB_SOURCES`:

- `raw_customer_service` — `raw/客服/bitget/*.md` (FAQ), `raw/客服/superex/*.csv` (9-lang scripts)
- `raw_official_tutorial` — `raw/官方教程/` (MD + HTML + PDF; images skipped)
- `wiki` — entire `wiki/` (frontmatter parsed; type/audience/aliases as metadata)

Rebuild after adding new sources:

```bash
python -m scripts.rebuild_index --force
# or just one source
python -m scripts.rebuild_index --force --source raw_customer_service
```

## Tuning knobs

`config.py`:

- `SIM_THRESHOLD` (0.45) — below this, refuse.
- `TOP_K` (5)
- `HYBRID_VECTOR_WEIGHT` (0.7) — vector vs BM25 mix.
- `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS`.
- `ENABLE_WEAK_TRIGGER` (False in P0) — turn on once router accuracy is tuned.

Environment overrides:

- `AGENT_BOT_ALLOWED_CHATS=-100123,-100456` (whitelist)
- `AGENT_BOT_ADMINS=12345,67890`

## P0 → P1 → P2

- **P0**: `/ask` + @mention + private chat; CSV+bitget MD only is enough to demo.
- **P1**: PDF + wiki + reply-chain multi-turn + weak trigger + DeepSeek fallback.
- **P2**: reranker, OCR for scanned PDFs, admin commands, eval harness over `qa_log.jsonl`.
