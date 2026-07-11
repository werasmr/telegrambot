"""
news_engine.py — модуль сбора свежих финансовых новостей из RSS-лент.

Используются публичные бесплатные RSS-ленты крупных финансовых изданий
(Yahoo Finance, Investing.com, CNBC, MarketWatch, FXStreet, ForexLive,
Bloomberg; для крипты — CoinDesk, Cointelegraph, Decrypt и др.).
Ключи API не требуются. Ленты качаются параллельно; недоступные
источники молча пропускаются.
"""

from __future__ import annotations

import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import feedparser

# Общерыночные ленты (форекс, макроэкономика, рынки): (название, URL)
GENERAL_FEEDS: list[tuple[str, str]] = [
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("Investing.com Forex", "https://www.investing.com/rss/news_1.rss"),
    ("Investing.com Economy", "https://www.investing.com/rss/news_95.rss"),
    ("Investing.com Economic Indicators", "https://www.investing.com/rss/news_25.rss"),
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("MarketWatch Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("MarketWatch Real-time", "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
    ("FXStreet", "https://xml.fxstreet.com/news/forex-news/index.xml"),
    ("FXEmpire", "https://www.fxempire.com/api/v1/en/articles/rss/news"),
    ("ForexLive", "https://www.forexlive.com/feed/news"),
    ("Investing.com Stock Market", "https://www.investing.com/rss/news_14.rss"),
    ("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss"),
    ("Bloomberg Economics", "https://feeds.bloomberg.com/economics/news.rss"),
]

# Ленты для криптовалютных активов
CRYPTO_FEEDS: list[tuple[str, str]] = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed"),
    ("CryptoPotato", "https://cryptopotato.com/feed/"),
    ("Investing.com Crypto", "https://www.investing.com/rss/news_301.rss"),
]

# Таймаут на скачивание одной ленты, секунд
FEED_TIMEOUT = 8

# Сколько лент качать одновременно
MAX_WORKERS = 8

# Заголовок User-Agent — часть лент отдаёт 403 без «браузерного» агента
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# Ключевые слова релевантности: новости с этими словами в заголовке
# поднимаются в топ для соответствующего актива.
ASSET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "EUR/USD": (
        "eur", "euro", "eurozone", "ecb", "lagarde", "dollar", "usd",
        "fed", "fomc", "powell", "inflation", "cpi", "rate", "rates",
        "treasury", "nonfarm", "payroll", "payrolls", "gdp", "forex",
    ),
    "GBP/USD": (
        "gbp", "pound", "sterling", "cable", "boe", "bank of england",
        "uk economy", "britain", "dollar", "usd", "fed", "fomc", "powell",
        "inflation", "cpi", "rate", "rates", "nonfarm", "payroll",
        "payrolls", "gdp", "forex",
    ),
    "BTC/USD": (
        "btc", "bitcoin", "crypto", "cryptocurrency", "ethereum", "eth",
        "blockchain", "stablecoin", "etf", "sec", "halving", "miner",
        "binance", "coinbase", "defi", "altcoin",
    ),
}


def _parse_feed(name: str, url: str) -> list[dict]:
    """Скачивает и разбирает одну RSS-ленту. При ошибке возвращает []."""
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(FEED_TIMEOUT)
    try:
        feed = feedparser.parse(url, agent=USER_AGENT)
    except Exception:
        return []
    finally:
        socket.setdefaulttimeout(old_timeout)

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
                "source": name,
                "published_ts": published_ts,
                "published": datetime.fromtimestamp(published_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                if published_ts
                else "неизвестно",
            }
        )
    return items


def _is_relevant(title: str, asset: str) -> bool:
    """Проверяет, содержит ли заголовок ключевые слова по активу (по границам слов)."""
    keywords = ASSET_KEYWORDS.get(asset, ())
    lower = title.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in keywords)


def fetch_latest_news(asset: str = "", limit: int = 5) -> list[dict]:
    """
    Возвращает список последних новостей (отсортирован от свежих к старым).
    Ленты качаются параллельно, дубликаты заголовков убираются.
    Новости, релевантные активу (по ключевым словам), поднимаются в топ.

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
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_parse_feed, name, url) for name, url in feeds]
        for future in as_completed(futures):
            all_items.extend(future.result())

    seen_titles: set[str] = set()
    unique_items: list[dict] = []
    for item in sorted(all_items, key=lambda x: x["published_ts"], reverse=True):
        key = item["title"].lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique_items.append(item)

    # Сначала — релевантные активу заголовки (свежие впереди),
    # затем добираем общими рыночными новостями до лимита.
    relevant = [i for i in unique_items if _is_relevant(i["title"], asset)]
    other = [i for i in unique_items if not _is_relevant(i["title"], asset)]
    return (relevant + other)[:limit]


def headlines_for_prompt(news: list[dict]) -> str:
    """Форматирует заголовки новостей в текстовый блок для промпта ИИ."""
    if not news:
        return "Свежие новости получить не удалось (лента недоступна)."
    lines = [
        f"{i}. [{item['published']}] ({item['source']}) {item['title']}"
        for i, item in enumerate(news, start=1)
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Быстрая ручная проверка модуля: python news_engine.py
    for asset_name in ("EUR/USD", "BTC/USD"):
        print(f"=== Новости для {asset_name} ===")
        print(headlines_for_prompt(fetch_latest_news(asset_name, limit=8)))
        print()
