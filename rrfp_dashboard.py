import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from streamlit_autorefresh import st_autorefresh
import textwrap

st.set_page_config(page_title="RRFP — Dashboard", layout="wide")
st.image("assets/logo.png", width=220)
st.title("Дашборд по опросу участников конференции")

st_autorefresh(interval=300 * 1000, key="datarefresh")

SHEET_CSV_URL = st.secrets.get("SHEET_CSV_URL", "")

@st.cache_data(ttl=300)
def load_data_from_gsheets(csv_url: str) -> pd.DataFrame:
    return pd.read_csv(csv_url)

def split_multi(series):
    vals = []
    for v in series.dropna():
        vals.extend([x.strip() for x in str(v).split(",")])
    return pd.Series([x for x in vals if x])

def barh_pretty(counts: pd.Series, title: str, xlabel: str = "Количество", wrap: int = 28):
    if counts is None or len(counts) == 0:
        st.info("Недостаточно данных для графика.")
        return

    labels = [textwrap.fill(str(x), wrap) for x in counts.index.tolist()]
    values = counts.values.tolist()

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5 * len(labels))))
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)

    max_v = max(values) if values else 0
    pad = max(0.02 * max_v, 0.1)
    for i, v in enumerate(values):
        ax.text(v + pad, i, str(v), va="center", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

if not SHEET_CSV_URL:
    st.error("Не задан SHEET_CSV_URL в Secrets Streamlit Cloud.")
    st.stop()

df = load_data_from_gsheets(SHEET_CSV_URL)
df.columns = [c.strip() for c in df.columns]


# ======== ФИЛЬТРЫ ========
st.sidebar.header("Фильтры")
attend_filter = st.sidebar.multiselect(
    "Посещение конференции",
    df["Посетил / Не посетил"].unique().tolist(),
    default=df["Посетил / Не посетил"].unique().tolist()
)

df_filtered = df[df["Посетил / Не посетил"].isin(attend_filter)]

# ======== 1. ОБЩАЯ СТАТИСТИКА ========
st.header("1. Общая статистика")

total = len(df_filtered)
st.write(f"Всего ответов после фильтрации: **{total}**")

st.write("### Посетили конференцию:")
barh_pretty(
df_filtered["Посетил / Не посетил"].value_counts(),
"Посещали ли конференцию в 2025 году",
xlabel="Ответов",
wrap=18
)


# ======== 2. ПОЛЕЗНЫЕ ФОРМАТЫ ========
st.header("2. Полезные форматы")

formats = split_multi(visited["Полезные форматы (если посещал)"])
barh_pretty(
    formats.value_counts(),
    "Какие форматы были наиболее полезны (по ответам посетивших)",
    xlabel="Упоминаний",
    wrap=28
)


# ======== 3. МАТРИЦА ОЦЕНОК ========
st.header("3. Оценка конференции (матрица)")

mean_scores = df_filtered[aspects].mean().sort_values(ascending=True)
barh_pretty(
    mean_scores,
    "Средняя оценка по аспектам (1–5)",
    xlabel="Средний балл",
    wrap=35
)


# ======== 4. ПРИЧИНЫ НЕУЧАСТИЯ ========
st.header("4. Причины неучастия")

reasons = split_multi(not_attended["Причины неучастия (если не посещал)"])
barh_pretty(
    reasons.value_counts(),
    "Причины неучастия в конференции 2025",
    xlabel="Упоминаний",
    wrap=38
)

# ======== 5. ТЕМЫ 2026 (WordCloud) ========
st.header("5. Предлагаемые темы 2026")

themes = " ".join(df_filtered["Темы 2026"].dropna().astype(str))

if themes.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(themes)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Ключевые слова")
    words = [w for w in themes.lower().split() if len(w) > 3]
    freq = Counter(words).most_common(20)
    st.write(freq)

# ======== 6. ДОПОЛНИТЕЛЬНЫЕ ПРЕДЛОЖЕНИЯ ========
st.header("6. Дополнительные предложения")

suggestions = " ".join(df_filtered["Доп. предложения"].dropna().astype(str))
if suggestions.strip():
    wc2 = WordCloud(width=800, height=400, background_color="white").generate(suggestions)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wc2, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)
