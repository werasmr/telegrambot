"""
ai_engine.py — ИИ-аналитик на базе бесплатного Google Gemini API.

Модель 'gemini-2.5-flash' получает сводку технических индикаторов и
последние заголовки новостей, взвешивает их и возвращает строгое решение:
"ВВЕРХ", "ВНИЗ" или "СТОИМ НА МЕСТЕ" с процентом уверенности,
рекомендованным временем экспирации и кратким обоснованием.

Ключ берётся из переменной окружения GEMINI_API_KEY (см. README.md)
или передаётся явно в get_ai_signal(api_key=...).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"

VALID_SIGNALS = ("ВВЕРХ", "ВНИЗ", "СТОИМ НА МЕСТЕ")


class AIEngineError(Exception):
    """Ошибка обращения к Gemini или разбора его ответа."""


@dataclass
class AISignal:
    """Итоговое решение ИИ-аналитика."""

    signal: str               # "ВВЕРХ" | "ВНИЗ" | "СТОИМ НА МЕСТЕ"
    confidence: int           # уверенность в процентах, 0–100
    expiration_minutes: int   # рекомендованная экспирация, минут (1–5)
    reasoning: str            # краткое обоснование на русском языке


SYSTEM_INSTRUCTION = """Ты — опытный трейдер-аналитик бинарных опционов.
Твоя задача — по краткосрочным техническим индикаторам и свежему новостному
фону дать строгий торговый сигнал на ближайшие 1–5 минут.

Правила анализа:
1. Технический анализ первичен (вес ~70%), новостной фон вторичен (вес ~30%),
   но резкие новости по активу могут перевесить технику.
2. RSI < 30 — перепроданность (аргумент за ВВЕРХ), RSI > 70 — перекупленность
   (аргумент за ВНИЗ). RSI в зоне 45–55 — нейтрально.
3. Пересечение MACD выше сигнальной линии и рост гистограммы — бычий сигнал;
   ниже сигнальной и падение гистограммы — медвежий.
4. Цена у нижней полосы Боллинджера (позиция < 0.15) — вероятен отскок вверх;
   у верхней (позиция > 0.85) — отскок вниз.
5. EMA9 > EMA21 — краткосрочный тренд вверх; EMA9 < EMA21 — вниз.
6. Если сигналы индикаторов противоречат друг другу и явного перевеса нет —
   выбирай "СТОИМ НА МЕСТЕ" с низкой уверенностью. Не выдумывай сигнал.
7. Чем сильнее совпадение сигналов, тем выше уверенность. Уверенность выше 85
   ставь только при полном совпадении всех индикаторов и новостного фона.
8. Экспирацию выбирай от 1 до 5 минут: короче — для импульсных движений,
   длиннее — для трендовых.

Отвечай СТРОГО в формате JSON без каких-либо пояснений вне JSON:
{
  "signal": "ВВЕРХ" | "ВНИЗ" | "СТОИМ НА МЕСТЕ",
  "confidence": целое число от 0 до 100,
  "expiration_minutes": целое число от 1 до 5,
  "reasoning": "краткое обоснование на русском языке, 2-4 предложения"
}
"""


def build_prompt(market_summary: str, news_block: str) -> str:
    """Собирает пользовательский промпт из тех. сводки и новостей."""
    return (
        "ДАННЫЕ ТЕХНИЧЕСКОГО АНАЛИЗА (последняя закрытая свеча):\n"
        f"{market_summary}\n\n"
        "ПОСЛЕДНИЕ 5 ЗАГОЛОВКОВ ФИНАНСОВЫХ НОВОСТЕЙ:\n"
        f"{news_block}\n\n"
        "Взвесь эти данные и выдай торговое решение в требуемом JSON-формате."
    )


def _extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (в т.ч. из markdown-блока)."""
    text = text.strip()
    # Убираем возможную обёртку ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIEngineError(f"Не удалось разобрать JSON из ответа модели: {text!r}") from exc


def _validate(data: dict) -> AISignal:
    """Проверяет и нормализует ответ модели."""
    signal = str(data.get("signal", "")).strip().upper()
    if signal not in VALID_SIGNALS:
        raise AIEngineError(f"Модель вернула недопустимый сигнал: {signal!r}")

    try:
        confidence = int(round(float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))

    try:
        expiration = int(round(float(data.get("expiration_minutes", 3))))
    except (TypeError, ValueError):
        expiration = 3
    expiration = max(1, min(5, expiration))

    reasoning = str(data.get("reasoning", "")).strip() or "Обоснование не предоставлено."

    return AISignal(
        signal=signal,
        confidence=confidence,
        expiration_minutes=expiration,
        reasoning=reasoning,
    )


def get_ai_signal(
    market_summary: str,
    news_block: str,
    api_key: str | None = None,
) -> AISignal:
    """
    Отправляет данные в Gemini и возвращает торговый сигнал.

    :param market_summary: текстовая сводка индикаторов (data_engine.get_market_snapshot()["summary"]).
    :param news_block: блок заголовков новостей (news_engine.headlines_for_prompt()).
    :param api_key: ключ Gemini; если не задан — берётся из окружения GEMINI_API_KEY.
    """
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise AIEngineError(
            "Не задан GEMINI_API_KEY. Получите бесплатный ключ на "
            "https://aistudio.google.com/apikey и укажите его "
            "(см. README.md)."
        )

    client = genai.Client(api_key=key)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_prompt(market_summary, news_block),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise AIEngineError(f"Ошибка запроса к Gemini API: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise AIEngineError("Gemini вернул пустой ответ (возможно, сработал фильтр безопасности).")

    return _validate(_extract_json(text))


if __name__ == "__main__":
    # Быстрая ручная проверка модуля (нужен GEMINI_API_KEY в окружении):
    # python ai_engine.py
    demo_summary = (
        "Актив: EUR/USD, таймфрейм свечей: 5m.\n"
        "Текущая цена: 1.085000 (изменение за последнюю свечу: +0.0500%).\n"
        "RSI(14): 26.5 (<30 — перепроданность, >70 — перекупленность).\n"
        "MACD: -0.00021, сигнальная линия: -0.00035, гистограмма: 0.00014 "
        "(гистограмма MACD растёт).\n"
        "Полосы Боллинджера: нижняя 1.0845, средняя 1.0862, верхняя 1.0879. "
        "Позиция цены внутри полос: 0.09 (0 — у нижней полосы, 1 — у верхней).\n"
        "EMA(9): 1.08505, EMA(21): 1.08490 — тренд бычий (EMA9 > EMA21)."
    )
    demo_news = (
        "1. [2026-01-01 10:00 UTC] ECB keeps rates unchanged, signals patience\n"
        "2. [2026-01-01 09:40 UTC] Dollar slips as risk appetite improves\n"
        "3. [2026-01-01 09:10 UTC] Eurozone PMI beats expectations\n"
        "4. [2026-01-01 08:50 UTC] US futures point to a flat open\n"
        "5. [2026-01-01 08:20 UTC] Oil steadies after volatile session"
    )
    result = get_ai_signal(demo_summary, demo_news)
    print(result)
