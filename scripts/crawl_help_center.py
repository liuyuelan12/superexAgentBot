"""Crawl the SuperEx Help Center into raw/帮助中心/ as markdown + a manifest.

Announcements are excluded by category id (see kb.zendesk): they are 95% of the
Help Center and are listing/maintenance notices, which would drown the rule and
tutorial content the support bot actually needs.

Usage:
    python -m scripts.crawl_help_center                       # all locales
    python -m scripts.crawl_help_center --locale zh-hk --locale en-001
    python -m scripts.crawl_help_center --dry-run             # counts only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HELP_CENTER_DIR, OUTPUT_DIR  # noqa: E402
from kb.zendesk import Article, fetch_locales, fetch_rule_articles  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s", level=logging.INFO
)
logger = logging.getLogger("crawl_help_center")

MANIFEST_PATH = OUTPUT_DIR / "superex-help-center-sitemap.json"
SITEMAP_MD_PATH = OUTPUT_DIR / "superex-help-center-sitemap.md"


def _safe_dirname(name: str) -> str:
    cleaned = "".join(c for c in name if c not in '/\\:*?"<>|').strip()
    return cleaned or "misc"


def _yaml_escape(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_markdown(article: Article) -> str:
    """Frontmatter carries provenance so conflict resolution can compare dates."""
    labels = ", ".join(_yaml_escape(x) for x in article.labels)
    return (
        "---\n"
        f"title: {_yaml_escape(article.title)}\n"
        f"article_id: {article.id}\n"
        f"url: {_yaml_escape(article.url)}\n"
        f"locale: {article.locale}\n"
        f"lang: {article.lang}\n"
        f"category: {_yaml_escape(article.category)}\n"
        f"section: {_yaml_escape(article.section)}\n"
        f"created_at: {_yaml_escape(article.created_at)}\n"
        f"updated_at: {_yaml_escape(article.updated_at)}\n"
        f"edited_at: {_yaml_escape(article.edited_at)}\n"
        f"labels: [{labels}]\n"
        "source: superex_help_center\n"
        "---\n\n"
        f"# {article.title}\n\n"
        f"{article.body_markdown}\n\n"
        f"来源: {article.url}\n"
    )


def write_articles(articles: list[Article], root: Path) -> list[dict]:
    manifest: list[dict] = []
    for art in articles:
        folder = root / art.locale / _safe_dirname(art.category)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{art.id}-{art.slug()}.md"
        path.write_text(render_markdown(art), encoding="utf-8")
        manifest.append(
            {
                "url": art.url,
                "title": art.title,
                "category": art.category,
                "section": art.section,
                "locale": art.locale,
                "lang": art.lang,
                "requires_login": False,
                "created_at": art.created_at,
                "updated_at": art.updated_at,
                "article_id": art.id,
                "path": str(path.relative_to(root.parent.parent)),
                "chars": len(art.body_markdown),
            }
        )
    return manifest


def write_sitemap_markdown(manifest: list[dict], path: Path) -> None:
    by_locale: dict[str, list[dict]] = {}
    for row in manifest:
        by_locale.setdefault(row["locale"], []).append(row)
    lines = [
        "# SuperEx 帮助中心 URL 清单",
        "",
        f"> 抓取自 support.superex.com Help Center API,共 {len(manifest)} 篇规则/教程类文章。",
        "> 公告类(上币/下币/维护/活动)按约定未纳入。",
        "",
    ]
    for locale in sorted(by_locale):
        rows = sorted(by_locale[locale], key=lambda r: (r["category"], r["title"]))
        lines += [f"## {locale} ({len(rows)} 篇)", "", "| 标题 | 分类 | 板块 | 更新时间 | URL |", "|---|---|---|---|---|"]
        for r in rows:
            title = r["title"].replace("|", "\\|")
            lines.append(
                f"| {title} | {r['category']} | {r['section']} | {r['updated_at'][:10]} | {r['url']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl SuperEx Help Center")
    parser.add_argument("--locale", action="append", help="Repeatable; default all")
    parser.add_argument("--dry-run", action="store_true", help="Count only, write nothing")
    args = parser.parse_args()

    locales = args.locale or fetch_locales()
    logger.info("Crawling %d locales: %s", len(locales), ", ".join(locales))

    all_articles: list[Article] = []
    for locale in locales:
        try:
            all_articles.extend(fetch_rule_articles(locale))
        except Exception as exc:  # noqa: BLE001 - one bad locale must not kill the run
            logger.error("locale %s failed: %s", locale, exc)

    logger.info("Fetched %d rule articles total", len(all_articles))
    if args.dry_run:
        by_locale: dict[str, int] = {}
        for a in all_articles:
            by_locale[a.locale] = by_locale.get(a.locale, 0) + 1
        for loc, n in sorted(by_locale.items()):
            logger.info("  %s: %d", loc, n)
        return 0

    HELP_CENTER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = write_articles(all_articles, HELP_CENTER_DIR)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_sitemap_markdown(manifest, SITEMAP_MD_PATH)
    logger.info("Wrote %d articles under %s", len(manifest), HELP_CENTER_DIR)
    logger.info("Manifest: %s", MANIFEST_PATH)
    logger.info("Sitemap:  %s", SITEMAP_MD_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
