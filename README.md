# 🎬 Analisis & Prediksi Keuntungan dari Penjualan Film

> **Proyek Akhir Praktikum Data Science**

**Nomor Kelompok:** 18  
**Nama Kelompok:** Poco X3 Pro Mas Jaka

| No | Nama | NIM |
|----|------|-----|
| 1 | Alexander Ganendra Bagaskara | 123240097 |
| 2 | Rafi Dzaka Pratama Putra | 123240104 |
| 3 | Hisyam L Baihaqi | 123240117 |
| 4 | Raihan Natawangsa | 123240120 |

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk **menganalisis dan memprediksi pendapatan (revenue) film** berdasarkan berbagai faktor seperti budget, genre, durasi, bahasa, negara produksi, sutradara, dan pemeran utama. Proses analisis dilakukan mengikuti tahapan sains data secara lengkap mulai dari *business understanding*, *data gathering*, *EDA*, *data wrangling*, *feature engineering*, *feature selection*, *modelling*, hingga *deployment*.

Hasil akhir proyek ini adalah sebuah **aplikasi web interaktif berbasis Streamlit** yang memungkinkan pengguna untuk:
- **Memprediksi estimasi revenue** sebuah film berdasarkan parameter yang dimasukkan
- **Mendapatkan rekomendasi** kombinasi genre dan bulan rilis terbaik untuk memaksimalkan pendapatan
- **Mengeksplorasi tren pasar** film berdasarkan data historis

---

## 📁 Struktur File

```
├── README.md                   # Dokumentasi proyek
├── TMDB_all_movies.csv         # Dataset mentah (~1,26 juta baris)
├── data_bersih_tmdb.csv        # Dataset bersih hasil preprocessing (~12.901 baris)
├── Proyek_Akhir_coba.ipynb     # Notebook: EDA & Data Wrangling
├── Modelling.ipynb             # Notebook: Feature Selection & Modelling
├── model_tmdb.joblib           # Model terlatih (disimpan via joblib)
├── app.py                      # Aplikasi web Streamlit
└── requirements.txt            # Dependensi Python
```

---

## 📊 Dataset

| Atribut | Keterangan |
|---------|------------|
| **Nama** | The Ultimate 1Million Movies Dataset (TMDB Movies Daily Updates) |
| **Penyusun** | Alan Vourch |
| **Sumber** | [Kaggle - TMDB Movies Daily Updates](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates?resource=download) |
| **Lisensi** | Apache 2.0 License |
| **Format & Ukuran** | CSV, ~798 MB, ~1.000.000 rekaman film, 30 kolom |

Dataset ini mengagregasi metadata komprehensif dari The Movie Database (TMDB) API yang dikombinasikan dengan pengidentifikasi IMDb. Kolom yang digunakan mencakup indikator finansial (`budget`, `revenue`), indikator penerimaan khalayak (`vote_average`, `vote_count`, `popularity`), serta metadata sinematik (`genres`, `runtime`, `director`, dll).

**Alasan pemilihan:** Dataset menyediakan informasi multi-dimensi paling lengkap di industri film modern. Kondisi datanya yang mentah (*dirty data*) memenuhi syarat praktikum untuk tahap pembersihan, transformasi, dan prapemrosesan data.

---

## 🔄 Alur Kerja

### 1. Business Understanding
Merumuskan permasalahan: bagaimana memprediksi pendapatan film berdasarkan faktor-faktor produksi yang dapat dikontrol.

### 2. Data Gathering & EDA (`Proyek_Akhir_coba.ipynb`)
- Memuat dataset mentah (~1,26 juta baris, 30 kolom)
- Analisis missing value, anomali (budget/revenue = 0), dan duplikat
- Analisis univariat (distribusi budget, revenue, runtime, genre)
- Analisis bivariat (korelasi budget vs revenue, revenue per genre/bahasa/negara)
- Analisis korelasi dan deteksi outlier (IQR)
- Pengecekan bias representasi (bahasa & negara)

### 3. Data Wrangling & Feature Engineering (`Proyek_Akhir_coba.ipynb`)
- Filter: hanya film `Released` dengan budget ≥ 10.000, revenue ≥ 10.000, runtime 40–300 menit
- Transformasi logaritmik pada `budget` dan `revenue` (mengatasi skewness)
- Encoding bulan rilis dengan sin/cos (cyclical encoding)
- Grouping genre, bahasa, dan negara menjadi kategori utama + "Other"
- Menghitung pengalaman sutradara (jumlah film sebelumnya)
- One-hot encoding untuk fitur kategorikal
- Hasil: **12.901 baris** data bersih → `data_bersih_tmdb.csv`

### 4. Feature Selection & Modelling (`Modelling.ipynb`)
- **Filter:** Variance threshold, mutual information, F-test
- **Embedded:** Lasso, Random Forest feature importance, permutation importance
- **Wrapper:** RFECV (Recursive Feature Elimination with Cross-Validation)
- **Voting:** Fitur yang lolos ≥ 3 dari 6 metode dipilih
- **Model:** Gradient Boosting Regressor (dengan hyperparameter tuning via RandomizedSearchCV)
- **Target:** `log10(revenue)`
- **Metrik evaluasi:** RMSE (log) = **0.6528**

### 5. Deployment (`app.py`)
Aplikasi web Streamlit dengan 3 tab:
- **Prediksi:** Input parameter film → estimasi revenue + kisaran wajar + ROI
- **Rekomendasi:** Grid search semua kombinasi genre × bulan → top-10 + heatmap
- **Tren Pasar:** Visualisasi median revenue & ROI per genre, jumlah film per tahun

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10+
- pip

### Instalasi

```bash
# Clone repositori
git clone https://github.com/Hisyam-L/Data_Science_Analisis_Keuntungan_Dari_Penjualan_Film.git
cd Data_Science_Analisis_Keuntungan_Dari_Penjualan_Film

# Install dependensi
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### Menjalankan Notebook

Buka `Proyek_Akhir_coba.ipynb` (EDA & Wrangling) dan `Modelling.ipynb` (Feature Selection & Modelling) di Jupyter Notebook atau JupyterLab.

---

## 🛠️ Tech Stack

| Kategori | Library |
|----------|---------|
| Data Processing | `pandas`, `numpy` |
| Visualisasi | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Deployment | `streamlit` |
| Serialisasi Model | `joblib` |

---

## 📄 Lisensi

Dataset menggunakan lisensi **Apache 2.0**. Proyek ini dibuat untuk keperluan akademik (Praktikum Data Science).
