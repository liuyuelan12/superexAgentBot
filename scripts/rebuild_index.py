"""CLI to (re)build the hybrid index.

Usage:
    python -m scripts.rebuild_index --force
    python -m scripts.rebuild_index --source raw_customer_service
    python -m scripts.rebuild_index            # auto-detect drift
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import KB_SOURCES  # noqa: E402
from kb.indexer import build_index, needs_rebuild  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rebuild_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild SuperEx Agent Bot index")
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument(
        "--source",
        action="append",
        choices=list(KB_SOURCES.keys()),
        help="Limit to one or more sources (repeatable)",
    )
    args = parser.parse_args()

    sources = args.source or None

    if not args.force and not needs_rebuild(sources):
        logger.info("Index is up to date; pass --force to rebuild anyway")
        return 0

    meta = build_index(sources=sources, force=True)
    logger.info(
        "Built index with %d docs across %d files at %d",
        meta["doc_count"],
        len(meta["files"]),
        meta["built_at"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
