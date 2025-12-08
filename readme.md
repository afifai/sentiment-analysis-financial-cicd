# Vertex AI MLOps Challenge: Financial Sentiment Analysis

Repository ini adalah environment latihan untuk menerapkan MLOps sederhana menggunakan GitHub Actions dan Google Vertex AI.

Tujuan Anda: Membuat model Machine Learning terbaik untuk memprediksi sentimen berita finansial. Anda tidak perlu memusingkan infrastruktur atau deployment. Cukup kirim Pull Request dan CI/CD pipeline akan melakukan training serta evaluasi otomatis di cloud.

---

## Dataset Overview

Dataset yang digunakan adalah Financial Phrasebank. Data ini berisi kalimat-kalimat dari berita finansial yang telah dilabeli dengan sentimen: positive, neutral, atau negative.

### Lokasi Data
Data tersimpan secara terpusat di Google Cloud Storage. Pipeline training sudah dikonfigurasi untuk mengambil data dari path berikut:

`gs://<NAMA_BUCKET_ANDA>/data/sentiment-financial.csv`

### Sample Data (Top 5 Rows)

| Label | Text |
| :--- | :--- |
| neutral | According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing . |
| neutral | Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and telecommunications , the statement said . |
| negative | The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported . |
| positive | With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability . |
| positive | According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 percent -40 percent with an operating profit margin of 10 percent -20 percent of net sales . |

---

## Cara Berpartisipasi

Ikuti langkah-langkah berikut untuk mengirimkan model versi Anda.

### 1. Buat Branch Baru
Jangan bekerja di branch main. Buat branch baru dengan nama Anda atau nama tim.

```bash
git checkout -b feature/model-nama-anda
```

### 2. Edit Kode Training

Buka file src/train.py. Anda bebas melakukan eksperimen seperti:

- Feature Engineering. 
- Model Selection. 
- Hyperparameter Tuning.

Penting:
- Pastikan output metrics tetap disimpan ke metrics.json.
- Jangan menghapus logika inference pada bagian bawah script.

### 3. Commit, Push dan Pull Request

```bash
git add src/train.py
git commit -m "feat: improve model architecture"
git push origin feature/model-nama-anda
```

Buka GitHub lalu buat Pull Request ke branch main.

### 4. Tunggu Laporan Otomatis

Setelah PR dibuat, GitHub Action akan berjalan:

- Linting.
- Vertex AI Training.
- Reporting perbandingan metrics.

---

## Architecture dan Pipeline

Repository ini menggunakan arsitektur MLOps serverless.

- Trigger: Pull Request atau Push to Main.
- Orchestrator: GitHub Actions.
- Compute: Vertex AI Custom Training.
- Model Registry: Vertex AI Model Registry.
- Reporting: Script pembanding metrics.

---

## Struktur Folder

```
.
├── .github/workflows/
├── src/
│   ├── train.py
│   ├── submit_job.py
│   └── compare_metrics.py
├── requirements.txt
└── README.md
```

---

## Kriteria Evaluasi

- Accuracy.
- F1-Score per kelas.
- Inference check dari 5 kalimat sampel.

Selamat mencoba dan happy coding!
