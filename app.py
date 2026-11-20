import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Pendapatan Film", page_icon="🎬", layout="wide")

# ISI DENGAN RMSE (skala log10) dari hasil evaluasi test set di notebook Modelling.
# Dipakai untuk membuat rentang prediksi: 10**(pred ± RMSE_LOG)
RMSE_LOG = 0.75

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


@st.cache_resource
def load_model():
    return joblib.load("model_tmdb.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("data_bersih_tmdb.csv")


bundle = load_model()
df = load_data()
pre, ren, selected, model = bundle["pre"], bundle["rename"], bundle["selected"], bundle["model"]
INPUT_COLS = list(pre.feature_names_in_)


def build_row(budget, runtime, year, month, genre, lang, country,
              director="", actor="", director_prev=0):
    """Susun 1 baris input persis seperti X saat training."""
    row = {c: 0 for c in INPUT_COLS}
    row.update(
        log_budget=np.log10(budget),
        runtime=runtime,
        release_year=year,
        log_director_prev_films=np.log1p(director_prev),
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
        director_known=int(bool(director)),
        actor_known=int(bool(actor)),
        primary_director=director or "Unknown",
        lead_actor=actor or "Unknown",
    )
    for prefix, val in [("genre_grp_", genre), ("language_grp_", lang), ("country_grp_", country)]:
        if prefix + val in row:
            row[prefix + val] = 1
    return pd.DataFrame([row])[INPUT_COLS]


def predict_revenue(X):
    """Kembalikan prediksi revenue dalam USD (model memprediksi log10(revenue))."""
    enc = pre.transform(X).rename(columns=ren)
    return 10 ** model.predict(enc[selected])


def juta(x):
    return f"${x / 1e6:,.1f} juta"


genres = sorted(df["genre_grp"].unique())
langs = sorted(df["language_grp"].unique())
countries = sorted(df["country_grp"].unique())

st.title("🎬 Prediksi & Rekomendasi Pendapatan Film")
st.caption("Model dilatih dari data TMDB. Hasil berupa estimasi berdasarkan pola historis, bukan jaminan.")

tab1, tab2, tab3 = st.tabs(["Prediksi", "Rekomendasi", "Tren Pasar"])

# ---------------- TAB 1: PREDIKSI ----------------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        budget = st.number_input("Budget (USD)", min_value=10_000, value=20_000_000, step=1_000_000)
        runtime = st.slider("Durasi (menit)", 40, 300, 110)
        year = st.number_input("Tahun rilis", 1990, 2035, 2027)
        month = st.selectbox("Bulan rilis", range(1, 13), format_func=lambda m: BULAN[m - 1], index=5)
    with c2:
        genre = st.selectbox("Genre utama", genres)
        lang = st.selectbox("Bahasa asli", langs, index=langs.index("en") if "en" in langs else 0)
        country = st.selectbox("Negara produksi", countries)
        director = st.text_input("Sutradara (opsional)")
        actor = st.text_input("Pemeran utama (opsional)")
        director_prev = st.number_input("Jumlah film sebelumnya dari sutradara", 0, 100, 0)

    if st.button("Prediksi", type="primary"):
        X = build_row(budget, runtime, year, month, genre, lang, country,
                      director.strip(), actor.strip(), director_prev)
        pred = float(predict_revenue(X)[0])
        low, high = pred / 10 ** RMSE_LOG, pred * 10 ** RMSE_LOG
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimasi revenue", juta(pred))
        m2.metric("Kisaran wajar", f"{juta(low)} – {juta(high)}")
        m3.metric("Estimasi ROI", f"{pred / budget:.1f}x")
        st.info("Kisaran dihitung dari RMSE model pada data uji. Film dengan faktor tak terukur "
                "(franchise, pemasaran, jumlah layar) bisa jauh di luar kisaran ini.")

# ---------------- TAB 2: REKOMENDASI ----------------
with tab2:
    st.write("Tentukan budget dan batasan produksi, lalu model mencoba semua kombinasi genre dan bulan rilis.")
    r1, r2, r3 = st.columns(3)
    rb = r1.number_input("Budget (USD)", 10_000, value=20_000_000, step=1_000_000, key="rb")
    rr = r1.slider("Durasi (menit)", 40, 300, 110, key="rr")
    ry = r2.number_input("Tahun rilis", 1990, 2035, 2027, key="ry")
    rl = r2.selectbox("Bahasa asli", langs, index=langs.index("en") if "en" in langs else 0, key="rl")
    rc = r3.selectbox("Negara produksi", countries, key="rc")

    if st.button("Cari kombinasi terbaik"):
        rows, meta = [], []
        for g in genres:
            for m in range(1, 13):
                rows.append(build_row(rb, rr, ry, m, g, rl, rc))
                meta.append((g, m))
        preds = predict_revenue(pd.concat(rows, ignore_index=True))
        res = pd.DataFrame(meta, columns=["Genre", "Bulan"])
        res["Revenue (USD)"] = preds

        top = res.sort_values("Revenue (USD)", ascending=False).head(10).copy()
        top["Bulan"] = top["Bulan"].map(lambda m: BULAN[m - 1])
        top["Estimasi"] = top["Revenue (USD)"].map(juta)
        top["ROI"] = (top["Revenue (USD)"] / rb).map(lambda x: f"{x:.1f}x")
        st.subheader("10 kombinasi teratas")
        st.dataframe(top[["Genre", "Bulan", "Estimasi", "ROI"]], hide_index=True, use_container_width=True)

        pivot = res.pivot(index="Genre", columns="Bulan", values="Revenue (USD)") / 1e6
        pivot.columns = BULAN
        st.subheader("Peta estimasi revenue (juta USD)")
        st.dataframe(pivot.style.background_gradient(cmap="YlGn").format("{:.1f}"),
                     use_container_width=True)
        st.caption("Selisih antar genre/bulan biasanya kecil dibanding pengaruh budget. "
                   "Baca sebagai petunjuk relatif, bukan angka pasti.")

# ---------------- TAB 3: TREN PASAR ----------------
with tab3:
    tahun_terakhir = int(df["release_year"].max())
    n = st.slider("Lihat berapa tahun terakhir", 3, 15, 5)
    min_film = st.slider("Minimal jumlah film per genre", 10, 200, 30)
    rec = df[df["release_year"] > tahun_terakhir - n].copy()
    rec["roi"] = rec["revenue"] / rec["budget"]

    g = rec.groupby("genre_grp").agg(
        jumlah_film=("id", "count"),
        median_revenue=("revenue", "median"),
        median_roi=("roi", "median"),
    )
    g = g[g["jumlah_film"] >= min_film]

    st.subheader(f"Genre berdasarkan median revenue ({tahun_terakhir - n + 1}–{tahun_terakhir})")
    st.bar_chart((g["median_revenue"] / 1e6).sort_values(ascending=False))
    st.subheader("Genre berdasarkan median ROI (revenue / budget)")
    st.bar_chart(g["median_roi"].sort_values(ascending=False))

    st.subheader("Jumlah film per tahun (apakah genre sedang ramai?)")
    pilih = st.multiselect("Genre", genres, default=list(g.sort_values("median_revenue").index[-3:]))
    if pilih:
        tren = (rec[rec["genre_grp"].isin(pilih)]
                .groupby(["release_year", "genre_grp"]).size().unstack(fill_value=0))
        st.line_chart(tren)
    st.caption("Dataset ini statis dan hanya memuat film dengan budget & revenue tercatat, "
               "jadi 'tren' di sini adalah tren historis, bukan tren real-time.")