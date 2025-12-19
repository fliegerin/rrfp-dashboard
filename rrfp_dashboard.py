import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from streamlit_autorefresh import st_autorefresh
import textwrap

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="RRFP — Dashboard", layout="wide")
st.image("assets/logo.png", width=220)
st.title("Дашборд по опросу участников конференции")

# автообновление страницы
st_autorefresh(interval=300 * 1000, key="datarefresh")  # каждые 5 минут

# -------------------- SETTINGS --------------------
SHEET_CSV_URL = st.secrets.get("SHEET_CSV_URL", "")

# -------------------- HELPERS --------------------
@st.cache_data(ttl=300)
def load_data_from_gsheets(csv_url: str) -> pd.DataFrame:
    return pd.read_csv(csv_url, encoding="utf-8-sig")

def split_multi(series: pd.Series) -> pd.Series:
    vals = []
    for v in series.dropna():
        vals.extend([x.strip() for x in str(v).split(",")])
    return pd.Series([x for x in vals if x])

def barh_pretty(counts: pd.Series, title: str, xlabel: str = "Количество", wrap: int = 28):
    if counts is None or len(counts) == 0:
        st.info("Недостаточно данных для графика.")
        return

    # если это числовой Series (например, mean_scores), index может быть не строкой
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
        ax.text(v + pad, i, str(round(v, 2)) if isinstance(v, (int, float)) else str(v),
                va="center", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

def parse_matrix_series(matrix_series: pd.Series) -> pd.DataFrame:
    """
    Парсим колонку матрицы вида:
    'Организация ...: 5 | Площадки ...: 4 | ...'
    Возвращаем DataFrame с отдельными колонками-аспектами.
    """
    rows = []
    for raw in matrix_series.fillna("").astype(str):
        if not raw.strip():
            rows.append({})
            continue

        parts = [p.strip() for p in raw.split("|")]
        d = {}
        for part in parts:
            if ":" not in part:
                continue
            name, score = part.rsplit(":", 1)   # важно: rsplit, чтобы не ломалось от двоеточий в тексте
            name = name.strip()
            score = score.strip()
            try:
                d[name] = float(score.replace(",", "."))
            except ValueError:
                continue
        rows.append(d)

    return pd.DataFrame(rows, index=matrix_series.index)

def safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

# -------------------- LOAD --------------------
if not SHEET_CSV_URL:
    st.error("Не задан SHEET_CSV_URL в Secrets Streamlit Cloud.")
    st.stop()

df = load_data_from_gsheets(SHEET_CSV_URL)
df.columns = [c.strip() for c in df.columns]

# базовая проверка минимально нужной колонки
if "Посетил / Не посетил" not in df.columns:
    st.error("В данных нет колонки «Посетил / Не посетил». Проверь структуру таблицы.")
    st.stop()

# -------------------- FILTERS --------------------
st.sidebar.header("Фильтры")

attend_filter = st.sidebar.multiselect(
    "Посещение конференции",
    sorted(df["Посетил / Не посетил"].dropna().unique().tolist()),
    default=sorted(df["Посетил / Не посетил"].dropna().unique().tolist())
)
st.sidebar.subheader("Фильтр «самые недовольные»")

only_unhappy = st.sidebar.checkbox("Показать только недовольных (по матрице)", value=False)
unhappy_threshold = st.sidebar.slider("Порог средней оценки", 1.0, 5.0, 3.0, 0.1)

df_filtered = df[df["Посетил / Не посетил"].isin(attend_filter)].copy()

# -------------------- 1. OVERVIEW --------------------
st.header("1. Общая статистика")

total = len(df_filtered)
st.write(f"Всего ответов после фильтрации: **{total}**")

barh_pretty(
    df_filtered["Посетил / Не посетил"].value_counts(),
    "Посещали ли конференцию в 2025 году",
    xlabel="Ответов",
    wrap=18
)

visited = df_filtered[df_filtered["Посетил / Не посетил"] == "Да"].copy()
not_attended = df_filtered[df_filtered["Посетил / Не посетил"] == "Нет"].copy()

# --- расчёт среднего балла по матрице для "Да" ---
matrix_col = "Матрица оценок (если посещал)"

visited_with_score = visited.copy()
visited_with_score["avg_score"] = pd.NA  # на случай пустого

if matrix_col in visited_with_score.columns:
    matrix_df = parse_matrix_series(visited_with_score[matrix_col])
    if matrix_df.shape[1] > 0:
        visited_with_score["avg_score"] = matrix_df.mean(axis=1)

# применяем фильтр "недовольных" только к посетившим (потому что матрица только у них)
if only_unhappy:
    visited_with_score = visited_with_score[
        pd.to_numeric(visited_with_score["avg_score"], errors="coerce") <= unhappy_threshold
    ].copy()

# обновим visited после фильтра (чтобы весь дашборд подстроился)
visited = visited_with_score.copy()

# -------------------- 2. USEFUL FORMATS (VISITED) --------------------
st.header("2. Полезные форматы (если посещали)")

useful_col = "Полезные форматы (если посещал)"
if not safe_col(df_filtered, useful_col):
    st.warning("Нет колонки «Полезные форматы (если посещал)» в данных.")
elif visited.empty:
    st.info("Пока нет ответов от посетивших конференцию (или фильтр их исключил).")
else:
    formats = split_multi(visited[useful_col])
    barh_pretty(
        formats.value_counts(),
        "Какие форматы были наиболее полезны (по ответам посетивших)",
        xlabel="Упоминаний",
        wrap=28
    )

# -------------------- 3. MATRIX --------------------
st.header("3. Оценка конференции (матрица)")

matrix_col = "Матрица оценок (если посещал)"
if not safe_col(df_filtered, matrix_col):
    st.warning("Нет колонки «Матрица оценок (если посещал)» в данных.")
elif visited.empty:
    st.info("Матрица есть, но сейчас нет выбранных ответов от посетивших конференцию.")
else:
    matrix_df = parse_matrix_series(visited[matrix_col])

    # если аспекты не нашлись (например, пусто)
    if matrix_df.shape[1] == 0:
        st.info("Матрица оценок пока пустая или не распознана.")
    else:
        # средние значения по аспектам
        mean_scores = matrix_df.mean().sort_values(ascending=True)

        barh_pretty(
            mean_scores,
            "Средняя оценка по аспектам (1–5)",
            xlabel="Средний балл",
            wrap=42
        )

        st.caption("Подсчёт: среднее значение по каждому аспекту среди ответов «Да» в выбранной фильтрации.")

st.header("3.1 Ответы с самыми низкими оценками (по матрице)")

if "avg_score" not in visited.columns or visited["avg_score"].isna().all():
    st.info("Пока нельзя посчитать среднюю оценку: нет данных матрицы.")
else:
    # сколько показывать
    top_n = st.slider("Сколько ответов показать", 5, 50, 15)

    unhappy_table = visited.sort_values("avg_score", ascending=True).head(top_n)

    st.write(f"Показано ответов: **{len(unhappy_table)}** (самые низкие avg_score)")
    show_cols = ["avg_score"]
    for c in ["Дата и время", "Актуальность тем", "Что улучшить", "Доп. предложения", "Имя", "Email"]:
        if c in visited.columns:
            show_cols.append(c)

    st.dataframe(unhappy_table[show_cols], use_container_width=True)
    st.caption("avg_score — средняя оценка по аспектам матрицы. Ниже = хуже.")



# -------------------- 4. REASONS (NOT ATTENDED) --------------------
st.header("4. Причины неучастия (если не посещали)")

reasons_col = "Причины неучастия (если не посещал)"
if not safe_col(df_filtered, reasons_col):
    st.warning("Нет колонки «Причины неучастия (если не посещал)» в данных.")
elif not_attended.empty:
    st.info("Пока нет ответов от тех, кто не посещал конференцию (или фильтр их исключил).")
else:
    reasons = split_multi(not_attended[reasons_col])
    if reasons.empty:
        st.info("Причины неучастия не заполнены.")
    else:
        barh_pretty(
            reasons.value_counts(),
            "Причины неучастия в конференции 2025",
            xlabel="Упоминаний",
            wrap=45
        )

# -------------------- 5. THEMES 2026 --------------------
st.header("5. Предлагаемые темы 2026")

themes_col = "Темы 2026"
if not safe_col(df_filtered, themes_col):
    st.warning("Нет колонки «Темы 2026» в данных.")
else:
    themes_text = " ".join(df_filtered[themes_col].dropna().astype(str))
    if not themes_text.strip():
        st.info("Пока нет текста для тем 2026.")
    else:
        wc = WordCloud(width=1000, height=450, background_color="white").generate(themes_text)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)

                # Вместо "ключевых слов" — полный список предложений по темам 2026
        st.subheader("Все предложения по темам 2026 (как есть)")

        themes_series = df_filtered[themes_col].dropna().astype(str)
        themes_series = themes_series[themes_series.str.strip().astype(bool)]

        q2 = st.text_input(
            "Поиск по темам 2026",
            placeholder="Например: ИИ, устойчивое развитие, регионы, ESG, кадры, образование..."
        )

        show2 = themes_series
        if q2 and q2.strip():
            qq = q2.strip().lower()
            show2 = show2[show2.str.lower().str.contains(qq)]

        st.write(f"Показано предложений: **{len(show2)}** из **{len(themes_series)}**")

        st.dataframe(
            pd.DataFrame({"Предложение по темам 2026": show2.tolist()}),
            use_container_width=True
        )

        # Если есть дата — покажем последние 10 предложений по темам
        if safe_col(df_filtered, "Дата и время"):
            st.subheader("Последние полученные предложения по темам 2026")
            tmp2 = df_filtered[["Дата и время", themes_col]].dropna()
            tmp2 = tmp2[tmp2[themes_col].astype(str).str.strip().astype(bool)]
            tmp2["Дата и время"] = pd.to_datetime(tmp2["Дата и время"], errors="coerce", dayfirst=True)
            tmp2 = tmp2.sort_values("Дата и время", ascending=False).head(10)
            st.dataframe(
                tmp2.rename(columns={themes_col: "Предложение по темам 2026"}),
                use_container_width=True
            )


# -------------------- 6. EXTRA SUGGESTIONS --------------------
st.header("6. Дополнительные предложения")

suggestions_col = "Доп. предложения"
if not safe_col(df_filtered, suggestions_col):
    st.warning("Нет колонки «Доп. предложения» в данных.")
else:
    # соберём непустые ответы
    sugg = df_filtered[suggestions_col].dropna().astype(str)
    sugg = sugg[sugg.str.strip().astype(bool)]

    if sugg.empty:
        st.info("Пока нет дополнительных предложений.")
    else:
        st.caption("Ниже — сводка по словам/темам и полный список предложений")

        # 6.1 — быстрые “ключевые слова” (частотность, без сложной лингвистики)
        st.subheader("Часто встречающиеся слова (черновая сводка)")
        text = " ".join(sugg.tolist()).lower()

        # минимальная чистка
        for ch in ",.!?;:\"«»()[]{}…—–/\\'":
            text = text.replace(ch, " ")

        tokens = [w for w in text.split() if len(w) > 4]
        # можно убрать самые “бесполезные” слова (можешь расширять)
        stop = set(["очень", "нужно", "сделать", "чтобы", "который", "будет", "можно", "пожалуйста", "спасибо", "вообще", "просто"])
        tokens = [w for w in tokens if w not in stop]

        top = Counter(tokens).most_common(20)
        st.dataframe(pd.DataFrame(top, columns=["Слово", "Частота"]), use_container_width=True)

        # 6.2 — “темы” по ключевым словам (простая группировка)
        st.subheader("Быстрая группировка по темам (эвристика)")
        categories = {
            "Организация / логистика": ["логист", "регистрац", "расписан", "тайминг", "площадк", "проход", "трансфер", "навигац"],
            "Коммуникации / рассылки": ["письм", "приглаш", "коммуникац", "анонс", "сайт", "telegram", "телеграм", "напомин"],
            "Программа / контент": ["тема", "трек", "секц", "доклад", "контент", "пленар", "воркшоп", "кругл", "кейсы"],
            "Нетворкинг": ["нетворк", "общен", "знакомств", "встреч", "комьюнит"],
            "Онлайн / трансляции": ["онлайн", "трансляц", "запись", "видео", "стрим"],
        }

        cat_counts = {k: 0 for k in categories}
        other = 0

        for s in sugg.tolist():
            s_low = s.lower()
            hit = False
            for cat, keys in categories.items():
                if any(k in s_low for k in keys):
                    cat_counts[cat] += 1
                    hit = True
            if not hit:
                other += 1

        cat_counts["Другое / без ключевых слов"] = other
        cat_series = pd.Series(cat_counts).sort_values(ascending=False)

        barh_pretty(cat_series, "К каким темам чаще относятся предложения", xlabel="Количество предложений", wrap=30)

        # 6.3 — сами предложения (читаемо, с поиском)
        st.subheader("Все предложения (как есть)")
        query = st.text_input("Поиск по предложениям", placeholder="Например: регистрация, онлайн, трансляция, нетворкинг...")

        show = sugg
        if query and query.strip():
            q = query.strip().lower()
            show = show[show.str.lower().str.contains(q)]

        st.write(f"Показано предложений: **{len(show)}** из **{len(sugg)}**")
        st.dataframe(pd.DataFrame({"Предложение": show.tolist()}), use_container_width=True)

        # бонус: показать 10 последних (если есть дата)
        if safe_col(df_filtered, "Дата и время"):
            st.subheader("Последние полученные предложения")
            tmp = df_filtered[[ "Дата и время", suggestions_col ]].dropna()
            tmp = tmp[tmp[suggestions_col].astype(str).str.strip().astype(bool)]
            tmp["Дата и время"] = pd.to_datetime(tmp["Дата и время"], errors="coerce", dayfirst=True)
            tmp = tmp.sort_values("Дата и время", ascending=False).head(10)
            st.dataframe(tmp.rename(columns={suggestions_col: "Предложение"}), use_container_width=True)
