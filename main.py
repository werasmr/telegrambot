"""
main.py — веб-интерфейс на Streamlit для генерации торговых сигналов
(ВВЕРХ / ВНИЗ) по бинарным опционам.

Запуск:  streamlit run main.py
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ai_engine import AIEngineError, AISignal, get_ai_signal
from data_engine import ASSETS, DataEngineError, get_market_snapshot
from news_engine import fetch_latest_news, headlines_for_prompt

# ----------------------------- Настройка страницы -----------------------------

st.set_page_config(
    page_title="AI Сигналы | Бинарные опционы",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .signal-card {
        border-radius: 16px;
        padding: 32px 16px;
        text-align: center;
        margin: 8px 0 16px 0;
    }
    .signal-up   { background: rgba(0, 200, 83, 0.12); border: 2px solid #00c853; }
    .signal-down { background: rgba(255, 23, 68, 0.12); border: 2px solid #ff1744; }
    .signal-flat { background: rgba(158, 158, 158, 0.12); border: 2px solid #9e9e9e; }
    .signal-arrow { font-size: 110px; line-height: 1; margin: 0; }
    .signal-title { font-size: 34px; font-weight: 800; margin: 6px 0 0 0; }
    .signal-sub   { font-size: 16px; opacity: 0.75; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Боковая панель --------------------------------

with st.sidebar:
    st.header("⚙️ Настройки")

    asset = st.selectbox("Актив", list(ASSETS.keys()), index=0)
    interval = st.radio(
        "Таймфрейм свечей",
        options=["1m", "5m"],
        index=1,
        horizontal=True,
        help="1m — минутные свечи (история за 1 день), 5m — пятиминутные (за 5 дней).",
    )

    st.divider()
    st.subheader("🔑 Gemini API")
    default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "GEMINI_API_KEY",
        value=default_key,
        type="password",
        help="Бесплатный ключ: https://aistudio.google.com/apikey. "
        "Можно также задать переменной окружения GEMINI_API_KEY (см. README.md).",
    )
    if not api_key:
        st.warning("Укажите GEMINI_API_KEY, иначе ИИ-аналитик не сможет работать.")

    st.divider()
    st.caption(
        "⚠️ Приложение носит исследовательский характер и не является "
        "финансовой рекомендацией. Торговля бинарными опционами связана "
        "с высоким риском потери средств."
    )

# --------------------------------- Заголовок -----------------------------------

st.title("📈 AI-генератор сигналов для бинарных опционов")
st.caption(
    f"Актив: **{asset}** · Таймфрейм: **{interval}** · "
    "Тех. анализ (RSI, MACD, Bollinger, EMA) + новостной фон + Google Gemini Flash"
)

get_signal_clicked = st.button(
    "🚀 ПОЛУЧИТЬ СИГНАЛ",
    type="primary",
    use_container_width=True,
)

# ------------------------------ Вспомогательные --------------------------------


def render_signal_card(result: AISignal) -> None:
    """Рисует большую карточку сигнала со стрелкой."""
    if result.signal == "ВВЕРХ":
        css, arrow, title, color = "signal-up", "▲", "ВВЕРХ · CALL", "#00c853"
    elif result.signal == "ВНИЗ":
        css, arrow, title, color = "signal-down", "▼", "ВНИЗ · PUT", "#ff1744"
    else:
        css, arrow, title, color = "signal-flat", "■", "СТОИМ НА МЕСТЕ", "#9e9e9e"

    st.markdown(
        f"""
        <div class="signal-card {css}">
            <p class="signal-arrow" style="color:{color};">{arrow}</p>
            <p class="signal-title" style="color:{color};">{title}</p>
            <p class="signal-sub">Рекомендованная экспирация: {result.expiration_minutes} мин ·
            Уверенность ИИ: {result.confidence}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart(candles, asset_name: str) -> None:
    """Свечной график Plotly с EMA, полосами Боллинджера, RSI и MACD."""
    df = candles.tail(120)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(f"{asset_name} — свечи и индикаторы", "RSI (14)", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Свечи",
            increasing_line_color="#00c853",
            decreasing_line_color="#ff1744",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_upper"], name="BB верх",
                   line=dict(color="rgba(120,144,156,0.7)", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["BB_lower"], name="BB низ",
                   line=dict(color="rgba(120,144,156,0.7)", width=1),
                   fill="tonexty", fillcolor="rgba(120,144,156,0.08)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA_9"], name="EMA 9",
                   line=dict(color="#ffb300", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA_21"], name="EMA 21",
                   line=dict(color="#29b6f6", width=1.5)),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
                   line=dict(color="#ab47bc", width=1.5)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#ff1744", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00c853", row=2, col=1)

    hist_colors = ["#00c853" if v >= 0 else "#ff1744" for v in df["MACD_hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_hist"], name="MACD гист.",
               marker_color=hist_colors),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                   line=dict(color="#29b6f6", width=1.2)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_signal"], name="Сигнальная",
                   line=dict(color="#ffb300", width=1.2)),
        row=3, col=1,
    )

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------ Основной сценарий ------------------------------

if get_signal_clicked:
    # 1. Технический анализ
    try:
        with st.spinner(f"Загружаю свечи {asset} ({interval}) и считаю индикаторы…"):
            snapshot = get_market_snapshot(asset, interval)
    except DataEngineError as exc:
        st.error(f"Ошибка получения рыночных данных: {exc}")
        st.stop()

    # 2. Новости
    with st.spinner("Собираю свежие финансовые новости…"):
        news = fetch_latest_news(asset, limit=5)
        news_block = headlines_for_prompt(news)

    # 3. ИИ-аналитик
    try:
        with st.spinner("Gemini взвешивает данные и принимает решение…"):
            result = get_ai_signal(snapshot["summary"], news_block, api_key=api_key)
    except AIEngineError as exc:
        st.error(f"Ошибка ИИ-аналитика: {exc}")
        st.stop()

    # 4. Вывод результата
    col_signal, col_details = st.columns([1, 1])

    with col_signal:
        render_signal_card(result)
        st.progress(result.confidence / 100.0, text=f"Уверенность ИИ: {result.confidence}%")

    with col_details:
        st.subheader("🧠 Обоснование ИИ")
        st.info(result.reasoning)
        if result.model:
            st.caption(f"Модель: `{result.model}`")

        ind = snapshot["indicators"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Цена", f"{snapshot['price']:.5f}", f"{ind['price_change_pct']:+.3f}%")
        m2.metric("RSI (14)", f"{ind['rsi_14']:.1f}")
        m3.metric("MACD гист.", f"{ind['macd_hist']:+.5f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("EMA 9", f"{ind['ema_9']:.5f}")
        m5.metric("EMA 21", f"{ind['ema_21']:.5f}")
        m6.metric("Позиция в BB", f"{ind['bb_position']:.2f}")

    # 5. График
    st.divider()
    render_chart(snapshot["candles"], asset)

    # 6. Новости и сырые данные
    st.divider()
    col_news, col_raw = st.columns([1, 1])

    with col_news:
        st.subheader("📰 Новости, учтённые ИИ")
        if news:
            for item in news:
                if item["link"]:
                    st.markdown(
                        f"- [{item['title']}]({item['link']})  \n"
                        f"  *{item['source']} · {item['published']}*"
                    )
                else:
                    st.markdown(f"- {item['title']}  \n  *{item['source']} · {item['published']}*")
        else:
            st.write("Свежие новости получить не удалось — решение принято только по тех. анализу.")

    with col_raw:
        st.subheader("📋 Сводка, отправленная в Gemini")
        st.code(snapshot["summary"] + "\n\nНОВОСТИ:\n" + news_block, language="text")
else:
    st.info(
        "Выберите актив и таймфрейм в боковой панели, укажите GEMINI_API_KEY "
        "и нажмите **«ПОЛУЧИТЬ СИГНАЛ»**. Приложение скачает свежие свечи, "
        "рассчитает RSI, MACD, Bollinger Bands и EMA, соберёт последние новости "
        "и передаст всё ИИ-аналитику Google Gemini Flash для принятия решения."
    )
