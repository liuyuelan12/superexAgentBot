"""把客服填好的 KB-gap CSV 转成 markdown,落到 raw/客服/补充/。

Usage:
    python -m scripts.ingest_cs_csv raw/客服/补充/kb_gap_filled.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from config import REPO_ROOT

_OUTPUT_DIR = REPO_ROOT / "raw" / "客服" / "补充"


def _clean_text(s: str) -> str:
    return (s or "").strip()


def _slug(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip())
    return re.sub(r"[/\\#?]", "-", s)


def _read_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open(encoding="utf-8") as f:
        return [[(c or "").strip() for c in row] for row in csv.reader(f)]


_EXPECTED_HEADER_KEYWORDS = ("问题", "答案", "客服")


def _strip_header(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    first = " ".join(rows[0])
    if any(k in first for k in _EXPECTED_HEADER_KEYWORDS):
        return rows[1:]
    return rows


def csv_to_markdown(csv_path: Path) -> str:
    rows = _strip_header(_read_rows(csv_path))
    if not rows:
        raise SystemExit(f"empty csv: {csv_path}")

    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    skipped = 0
    for row in rows:
        cells = row + [""] * max(0, 5 - len(row))
        category = _clean_text(cells[0]) or "未分类"
        question = _clean_text(cells[1])
        answer = _clean_text(cells[3])
        note = _clean_text(cells[4])
        if not question or not answer:
            skipped += 1
            continue
        grouped.setdefault(category, []).append(
            {"question": question, "answer": answer, "note": note}
        )

    today = datetime.utcnow().strftime("%Y-%m-%d")
    out = [
        f"# SuperEx 客服补充 FAQ（{today}）",
        "",
        "> 客服团队填写,补齐资料库中缺失的常见交易所问题。",
        "",
    ]
    total = 0
    for category, items in grouped.items():
        out.append(f"## {category}")
        out.append("")
        for item in items:
            out.append(f"### {item['question']}")
            out.append("")
            out.append(item["answer"])
            if item["note"]:
                out.append("")
                out.append(f"参考链接:{item['note']}")
            out.append("")
            total += 1

    print(
        f"converted {total} Q&A across {len(grouped)} categories "
        f"(skipped {skipped} empty rows)",
        file=sys.stderr,
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        type=Path,
        help="filled CSV (e.g. raw/客服/补充/kb_gap_filled.csv)",
    )
    ap.add_argument(
        "--name",
        default="customer-support-supplement",
        help="output markdown basename (no extension)",
    )
    args = ap.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    if not csv_path.exists():
        raise SystemExit(f"csv not found: {csv_path}")

    md = csv_to_markdown(csv_path)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"{_slug(args.name)}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
