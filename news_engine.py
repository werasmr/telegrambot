"""
news_engine.py — модуль сбора свежих финансовых новостей из RSS-лент.

Используются публичные бесплатные RSS-ленты (Yahoo Finance, Investing.com,
CoinDesk для крипты). Ключи API не требуются.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import feedparser

# Общерыночные ленты (форекс, макроэкономика)
GENERAL_FEEDS: list[str] = [
    "https://finance.yahoo.com/news/rssindex",
    "https://www.investing.com/rss/news_1.rss",   # Forex news
    "https://www.investing.com/rss/news_95.rss",  # Economy news
]

# Ленты для криптовалютных активов
CRYPTO_FEEDS: list[str] = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.investing.com/rss/news_301.rss",  # Cryptocurrency news
]

# Таймаут на скачивание одной ленты, секунд
FEED_TIMEOUT = 10


def _parse_feed(url: str) -> list[dict]:
    """Скачивает и разбирает одну RSS-ленту. При ошибке возвращает []."""
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    items: list[dict] = []
    for entry in getattr(feed, "entries", []):
        title = (entry.get("title") or "").strip()
        if not title:
            continue

        published_ts = 0.0
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed:
            try:
                published_ts = time.mktime(parsed)
            except (OverflowError, ValueError):
                published_ts = 0.0

        items.append(
            {
                "title": title,
                "link": (entry.get("link") or "").strip(),
                "source": url,
                "published_ts": published_ts,
                "published": datetime.fromtimestamp(published_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                if published_ts
                else "неизвестно",
            }
        )
    return items


def fetch_latest_news(asset: str = "", limit: int = 5) -> list[dict]:
    """
    Возвращает список последних новостей (отсортирован от свежих к старым).

    :param asset: имя актива ("EUR/USD", "GBP/USD", "BTC/USD").
                  Для крипты дополнительно подключаются крипто-ленты.
    :param limit: сколько новостей вернуть.
    :return: список словарей {"title", "link", "source", "published", "published_ts"}.
    """
    feeds = list(GENERAL_FEEDS)
    if "BTC" in asset.upper() or "CRYPTO" in asset.upper():
        # Для криптовалют крипто-новости важнее — ставим их в начало.
        feeds = CRYPTO_FEEDS + GENERAL_FEEDS

    all_items: list[dict] = []
    seen_titles: set[str] = set()

    for url in feeds:
        for item in _parse_feed(url):
            key = item["title"].lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            all_items.append(item)

    all_items.sort(key=lambda x: x["published_ts"], reverse=True)
    return all_items[:limit]


def headlines_for_prompt(news: list[dict]) -> str:
    """Форматирует заголовки новостей в текстовый блок для промпта ИИ."""
    if not news:
        return "Свежие новости получить не удалось (лента недоступна)."
    lines = [
        f"{i}. [{item['published']}] {item['title']}"
        for i, item in enumerate(news, start=1)
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Быстрая ручная проверка модуля: python news_engine.py
    for asset_name in ("EUR/USD", "BTC/USD"):
        print(f"=== Новости для {asset_name} ===")
        print(headlines_for_prompt(fetch_latest_news(asset_name, limit=5)))
        print()
