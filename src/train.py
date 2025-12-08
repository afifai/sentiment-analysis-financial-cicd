import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# --- KONFIGURASI ENV VERTEX AI ---
# AIP_MODEL_DIR diset otomatis oleh Vertex AI untuk output artifacts
MODEL_DIR = os.getenv('AIP_MODEL_DIR', '.')
DATA_FILE = 'sentiment-financial.csv'

# Helper untuk download data jika running di Cloud
def ensure_data_exists():
    if not os.path.exists(DATA_FILE):
        # Asumsi bucket dipassing via arg atau hardcoded logic, 
        # tapi untuk simplifikasi kita ambil dari env var jika ada, 
        # atau kita asumsikan pipeline submit sudah menyiapkan mekanisme download.
        # Disini kita pakai gsutil command sederhana karena container Vertex support gsutil
        bucket_name = os.getenv("GCS_BUCKET_NAME") # Akan diinject oleh submit_job.py
        if bucket_name:
            print(f"Downloading data from {bucket_name}...")
            os.system(f"gsutil cp {bucket_name}/data/{DATA_FILE} .")
        else:
            print("Warning: Bucket name not found, expecting local file.")

def train_and_evaluate():
    ensure_data_exists()
    
    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE, header=None, names=['label', 'text'])
    except FileNotFoundError:
        print(f"Error: File {DATA_FILE} not found!")
        sys.exit(1)
    
    # 2. Preprocessing & Splitting (AREA PESERTA BISA UBAH)
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Pipeline (AREA PESERTA BISA UBAH)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    # 3. Evaluasi Metrics
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    labels = model.classes_
    f1_scores = f1_score(y_test, y_pred, average=None, labels=labels)
    
    # Format metrics standar untuk report
    metrics = {
        "model_name": "LogisticRegression",
        "parameters": str(model.get_params()),
        "accuracy": acc,
        "f1_scores": {label: score for label, score in zip(labels, f1_scores)}
    }
    
    # 4. Inference 5 Kalimat Wajib (JANGAN DIHAPUS)
    test_sentences = [
        ("The company reported weaker demand in the last quarter which forced management to revise its revenue outlook downward.", "negative"),
        ("According to the latest update, retail sales remained stable and the firm does not expect major changes in its current operations.", "neutral"),
        ("With the recent investment in its production line, the company expects capacity to increase and strengthen its growth trajectory.", "positive"),
        ("The firm stated that market uncertainties have caused delays in its planned expansion for the upcoming year.", "negative"),
        ("The company announced the opening of a new regional office to broaden its presence in the Asian B2B market.", "positive")
    ]
    
    inference_results = []
    print("\nRunning Inference Checks...")
    for text, expected in test_sentences:
        vec_text = vectorizer.transform([text])
        pred = model.predict(vec_text)[0]
        inference_results.append({
            "text": text,
            "expected": expected,
            "predicted": pred,
            "match": bool(expected == pred)
        })

    final_output = {
        "metrics": metrics,
        "inference": inference_results
    }

    # 5. Save Artifacts
    print(f"Saving model artifacts...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    with open('metrics.json', 'w') as f:
        json.dump(final_output, f, indent=2)
        
    # Upload ke GCS (Penting untuk Vertex AI)
    os.system(f"gsutil cp model.joblib {MODEL_DIR}/model.joblib")
    os.system(f"gsutil cp vectorizer.joblib {MODEL_DIR}/vectorizer.joblib")
    os.system(f"gsutil cp metrics.json {MODEL_DIR}/metrics.json")
    
    print("Training job finished.")

if __name__ == '__main__':
    train_and_evaluate()
