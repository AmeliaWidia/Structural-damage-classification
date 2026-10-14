# Nepal Damage Classification

Project supervised machine learning untuk memprediksi `damage_grade` pada data bangunan. Metrik evaluasi utama adalah **Macro F1**. Tidak menggunakan model neural network.

## Struktur Repository

```text
nepal-damage-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── metadata.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model.ipynb
│   └── 03_submissions.ipynb
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── best_model.pkl
├── submissions/
│   └── submission.csv
└── reports/
    └── figures/
```

## Persiapan

1. Buat dan aktifkan virtual environment Python.
2. Instal dependency: `pip install -r requirements.txt`.
3. Terima aturan kompetisi Kaggle dan unduh file data yang diperlukan.
4. Letakkan file CSV di folder `data/`. Jangan commit file dataset.
5. Buka root repository di VS Code dan jalankan notebook secara berurutan.

## Workflow

- `01_eda.ipynb`: berisi informasi dataset, statistik deskriptif, missing values, duplikasi, distribusi target, distribusi fitur numerik, boxplot fitur numerik terhadap target, distribusi fitur kategorikal terhadap target, serta ringkasan EDA. Grafik disimpan di `reports/figures/`.
- `02_model.ipynb`: berisi split stratified, preprocessing dengan `Pipeline` dan `ColumnTransformer`, encoding target untuk XGBoost, tujuh baseline (LDA, SVM, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM), perbandingan Macro F1/accuracy/waktu training, classification report, confusion matrix setiap model, GridSearchCV hanya untuk baseline terbaik, evaluasi baseline vs tuning, confusion matrix final, dan penyimpanan model `best_model.pkl`.
- `03_submissions.ipynb`: memuat model, menyiapkan fitur test dengan urutan kolom model, prediksi, membuat submission mengikuti template, validasi kelas/kolom/baris, menyimpan CSV dan memeriksa isi CSV yang tersimpan.

## Scripts

`preprocess.py` berisi fungsi loading, pemisahan fitur-target, casting ID geografis, pembuatan preprocessor, dan normalisasi label. 
`train.py` berisi factory model, training/evaluasi baseline, dan helper GridSearchCV. 
`predict.py` berisi helper simpan/muat model, prediksi, pembuatan submission, validasi, dan penyimpanan CSV. 

## Interpretasi hasil

- Baseline terbaik: LightGBM
- Baseline Macro F1: 0.5214
- Final validation Macro F1: 0.5279
- Parameter terbaik: n_estimator: 150, num_leaves:50
- Skor Kaggle: 0.53149

