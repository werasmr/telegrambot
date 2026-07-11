"""
data_engine.py — модуль получения рыночных данных и расчёта технических индикаторов.

Источник данных: Yahoo Finance (библиотека yfinance).
Индикаторы: RSI(14), MACD(12, 26, 9), Bollinger Bands(20, 2), EMA(9), EMA(21).
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

# Доступные активы: человекочитаемое имя -> тикер Yahoo Finance
ASSETS: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "XRP/USD": "XRP-USD",
    "BNB/USD": "BNB-USD",
    "DOGE/USD": "DOGE-USD",
    "ADA/USD": "ADA-USD",
}

# Криптовалютные активы (торгуются 24/7, для них подключаются крипто-новости)
CRYPTO_ASSETS: frozenset[str] = frozenset(
    {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "DOGE/USD", "ADA/USD"}
)

# Допустимые интервалы свечей и соответствующая глубина истории,
# достаточная для расчёта индикаторов (MACD требует минимум ~35 свечей).
INTERVALS: dict[str, str] = {
    "1m": "1d",   # минутные свечи — история за 1 день
    "5m": "5d",   # 5-минутные свечи — история за 5 дней
}


class DataEngineError(Exception):
    """Ошибка получения или обработки рыночных данных."""


def fetch_candles(asset: str, interval: str = "5m") -> pd.DataFrame:
    """
    Скачивает свечи (OHLCV) для указанного актива.

    :param asset: имя актива из ASSETS, например "EUR/USD".
    :param interval: "1m" или "5m".
    :return: DataFrame с колонками Open, High, Low, Close, Volume
             и DatetimeIndex по возрастанию времени.
    """
    if asset not in ASSETS:
        raise DataEngineError(
            f"Неизвестный актив '{asset}'. Доступны: {', '.join(ASSETS)}"
        )
    if interval not in INTERVALS:
        raise DataEngineError(
            f"Неизвестный интервал '{interval}'. Доступны: {', '.join(INTERVALS)}"
        )

    ticker = ASSETS[asset]
    period = INTERVALS[interval]

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        raise DataEngineError(
            f"Yahoo Finance не вернул данные по {asset} ({ticker}). "
            "Возможно, рынок закрыт или проблема с сетью."
        )

    # yfinance для одного тикера может вернуть MultiIndex-колонки
    # вида (Close, EURUSD=X) — сводим их к плоским именам.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df.sort_index(inplace=True)

    if len(df) < 40:
        raise DataEngineError(
            f"Получено слишком мало свечей ({len(df)}) для расчёта индикаторов. "
            "Попробуйте другой интервал или актив."
        )

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame колонки с техническими индикаторами:
    RSI_14, MACD, MACD_signal, MACD_hist, BB_upper, BB_middle, BB_lower,
    EMA_9, EMA_21.
    """
    out = df.copy()
    close = out["Close"]

    rsi = RSIIndicator(close=close, window=14)
    out["RSI_14"] = rsi.rsi()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_hist"] = macd.macd_diff()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    out["BB_upper"] = bb.bollinger_hband()
    out["BB_middle"] = bb.bollinger_mavg()
    out["BB_lower"] = bb.bollinger_lband()

    out["EMA_9"] = EMAIndicator(close=close, window=9).ema_indicator()
    out["EMA_21"] = EMAIndicator(close=close, window=21).ema_indicator()

    return out


def get_market_snapshot(asset: str, interval: str = "5m") -> dict:
    """
    Полный цикл: скачать свечи, посчитать индикаторы и вернуть
    словарь с последними значениями — готовый вход для ИИ-аналитика.

    :return: {
        "asset", "interval", "price", "candles" (DataFrame),
        "indicators" (dict со значениями последней свечи),
        "summary" (готовая текстовая сводка для промпта),
    }
    """
    candles = add_indicators(fetch_candles(asset, interval))
    last = candles.iloc[-1]
    prev = candles.iloc[-2]

    price = float(last["Close"])
    price_change_pct = (price / float(prev["Close"]) - 1.0) * 100.0

    # Положение цены внутри полос Боллинджера: 0 — на нижней, 1 — на верхней.
    bb_range = float(last["BB_upper"]) - float(last["BB_lower"])
    bb_position = (price - float(last["BB_lower"])) / bb_range if bb_range > 0 else 0.5

    indicators = {
        "price": price,
        "price_change_pct": round(price_change_pct, 4),
        "rsi_14": round(float(last["RSI_14"]), 2),
        "macd": round(float(last["MACD"]), 6),
        "macd_signal": round(float(last["MACD_signal"]), 6),
        "macd_hist": round(float(last["MACD_hist"]), 6),
        "macd_hist_prev": round(float(prev["MACD_hist"]), 6),
        "bb_upper": round(float(last["BB_upper"]), 6),
        "bb_middle": round(float(last["BB_middle"]), 6),
        "bb_lower": round(float(last["BB_lower"]), 6),
        "bb_position": round(bb_position, 3),
        "ema_9": round(float(last["EMA_9"]), 6),
        "ema_21": round(float(last["EMA_21"]), 6),
        "ema_trend": "бычий (EMA9 > EMA21)"
        if float(last["EMA_9"]) > float(last["EMA_21"])
        else "медвежий (EMA9 < EMA21)",
    }

    summary = _build_summary(asset, interval, indicators)

    return {
        "asset": asset,
        "interval": interval,
        "price": price,
        "candles": candles,
        "indicators": indicators,
        "summary": summary,
    }


def quick_bias(ind: dict) -> dict:
    """
    Быстрый локальный анализ без ИИ: голосование индикаторов.

    Каждый индикатор голосует за рост (+1), падение (-1) или нейтрально (0).
    :return: {"label": "ВВЕРХ"|"ВНИЗ"|"НЕЙТРАЛЬНО", "score": -4..4, "reasons": [...]}
    """
    score = 0
    reasons: list[str] = []

    if ind["rsi_14"] < 30:
        score += 1
        reasons.append(f"RSI {ind['rsi_14']:.0f} — перепроданность")
    elif ind["rsi_14"] > 70:
        score -= 1
        reasons.append(f"RSI {ind['rsi_14']:.0f} — перекупленность")

    if ind["ema_9"] > ind["ema_21"]:
        score += 1
        reasons.append("EMA9 > EMA21 — тренд вверх")
    else:
        score -= 1
        reasons.append("EMA9 < EMA21 — тренд вниз")

    if ind["macd_hist"] > 0 and ind["macd_hist"] >= ind["macd_hist_prev"]:
        score += 1
        reasons.append("MACD выше сигнальной, импульс растёт")
    elif ind["macd_hist"] < 0 and ind["macd_hist"] <= ind["macd_hist_prev"]:
        score -= 1
        reasons.append("MACD ниже сигнальной, импульс падает")

    if ind["bb_position"] < 0.15:
        score += 1
        reasons.append("цена у нижней полосы Боллинджера")
    elif ind["bb_position"] > 0.85:
        score -= 1
        reasons.append("цена у верхней полосы Боллинджера")

    if score >= 2:
        label = "ВВЕРХ"
    elif score <= -2:
        label = "ВНИЗ"
    else:
        label = "НЕЙТРАЛЬНО"

    return {"label": label, "score": score, "reasons": reasons}


def _overview_row(asset: str, interval: str) -> dict:
    """Собирает строку обзора по одному активу (или запись об ошибке)."""
    try:
        candles = add_indicators(fetch_candles(asset, interval))
    except DataEngineError as exc:
        return {"asset": asset, "ok": False, "error": str(exc)}

    last = candles.iloc[-1]
    prev = candles.iloc[-2]
    price = float(last["Close"])

    # Изменение за последние 12 свечей (~1 час на таймфрейме 5m)
    lookback = min(12, len(candles) - 1)
    change_pct = (price / float(candles["Close"].iloc[-1 - lookback]) - 1.0) * 100.0

    bb_range = float(last["BB_upper"]) - float(last["BB_lower"])
    bb_position = (price - float(last["BB_lower"])) / bb_range if bb_range > 0 else 0.5

    ind = {
        "rsi_14": float(last["RSI_14"]),
        "ema_9": float(last["EMA_9"]),
        "ema_21": float(last["EMA_21"]),
        "macd_hist": float(last["MACD_hist"]),
        "macd_hist_prev": float(prev["MACD_hist"]),
        "bb_position": bb_position,
    }
    bias = quick_bias(ind)

    return {
        "asset": asset,
        "ok": True,
        "price": price,
        "change_pct": change_pct,
        "rsi": ind["rsi_14"],
        "ema_trend": "вверх" if ind["ema_9"] > ind["ema_21"] else "вниз",
        "bias": bias["label"],
        "score": bias["score"],
        "reasons": bias["reasons"],
        "is_crypto": asset in CRYPTO_ASSETS,
    }


def get_market_overview(interval: str = "5m") -> list[dict]:
    """
    Параллельно скачивает котировки по ВСЕМ активам и возвращает
    краткий анализ каждого: цена, изменение, RSI, тренд EMA и
    итоговый локальный вердикт (без обращения к ИИ).
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(ASSETS)) as pool:
        rows = list(pool.map(lambda a: _overview_row(a, interval), ASSETS))
    return rows


def _build_summary(asset: str, interval: str, ind: dict) -> str:
    """Формирует компактную текстовую сводку тех. анализа для промпта ИИ."""
    macd_direction = (
        "гистограмма MACD растёт"
        if ind["macd_hist"] > ind["macd_hist_prev"]
        else "гистограмма MACD падает"
    )
    lines = [
        f"Актив: {asset}, таймфрейм свечей: {interval}.",
        f"Текущая цена: {ind['price']:.6f} "
        f"(изменение за последнюю свечу: {ind['price_change_pct']:+.4f}%).",
        f"RSI(14): {ind['rsi_14']} "
        f"(<30 — перепроданность, >70 — перекупленность).",
        f"MACD: {ind['macd']}, сигнальная линия: {ind['macd_signal']}, "
        f"гистограмма: {ind['macd_hist']} ({macd_direction}).",
        f"Полосы Боллинджера: нижняя {ind['bb_lower']}, средняя {ind['bb_middle']}, "
        f"верхняя {ind['bb_upper']}. Позиция цены внутри полос: {ind['bb_position']} "
        f"(0 — у нижней полосы, 1 — у верхней).",
        f"EMA(9): {ind['ema_9']}, EMA(21): {ind['ema_21']} — тренд {ind['ema_trend']}.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Быстрая ручная проверка модуля: python data_engine.py
    for name in ASSETS:
        try:
            snapshot = get_market_snapshot(name, "5m")
            print(f"=== {name} ===")
            print(snapshot["summary"])
            print()
        except DataEngineError as exc:
            print(f"[{name}] ошибка: {exc}")
