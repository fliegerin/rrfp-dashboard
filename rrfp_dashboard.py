import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# ======== ЗАГРУЗКА ДАННЫХ ========
st.title("📊 Дашборд по опросу участников конференции РРФП")

uploaded = st.file_uploader("Загрузите CSV-файл с результатами опроса", type=["csv", "xlsx"])

if uploaded:
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")

    st.success(f"Файл загружен! Строк: {len(df)}")

    # Преобразуем заголовки
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
    st.bar_chart(df_filtered["Посетил / Не посетил"].value_counts())

    # ======== 2. ПОЛЕЗНЫЕ ФОРМАТЫ ========
    st.header("2. Полезные форматы")

    def split_multi(series):
        vals = []
        for v in series.dropna():
            vals.extend([x.strip() for x in str(v).split(",")])
        return pd.Series(vals)

    visited = df_filtered[df_filtered["Посетил / Не посетил"] == "Да"]
    if not visited.empty:
        st.subheader("Форматы, отмеченные как полезные")

        formats = split_multi(visited["Полезные форматы (если посещал)"])
        st.bar_chart(formats.value_counts())

    # ======== 3. МАТРИЦА ОЦЕНОК ========
    st.header("3. Оценка конференции (матрица)")

    if "Матрица оценок (если посещал)" in df.columns:
        def parse_matrix(row):
            if not isinstance(row, str): 
                return {}
            res = {}
            for part in row.split("|"):
                if ":" in part:
                    name, score = part.split(":")
                    res[name.strip()] = float(score.strip())
            return res

        matrix = visited["Матрица оценок (если посещал)"].apply(parse_matrix)
        aspects = sorted({k for d in matrix for k in d})

        for asp in aspects:
            df_filtered[asp] = matrix.apply(lambda d: d.get(asp, None))

        if aspects:
            st.bar_chart(df_filtered[aspects].mean())

    # ======== 4. ПРИЧИНЫ НЕУЧАСТИЯ ========
    st.header("4. Причины неучастия")

    not_attended = df_filtered[df_filtered["Посетил / Не посетил"] == "Нет"]
    if not not_attended.empty:
        reasons = split_multi(not_attended["Причины неучастия (если не посещал)"])
        st.bar_chart(reasons.value_counts())

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
