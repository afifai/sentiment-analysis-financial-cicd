import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# --- KONFIGURASI ENV VERTEX AI ---
MODEL_DIR = os.getenv('AIP_MODEL_DIR', '.')
DATA_FILE = 'sentiment-financial.csv'

def ensure_data_exists():
    """Download data dari GCS jika file lokal tidak ditemukan."""
    if not os.path.exists(DATA_FILE):
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            print("Warning: GCS_BUCKET_NAME variable not found. Expecting local file.")
            return

        print(f"Downloading data from {bucket_name}...")
        try:
            # Bersihkan prefix gs:// jika ada, karena client library tidak butuh itu
            bucket_name_clean = bucket_name.replace("gs://", "")
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name_clean)
            
            # Asumsi file ada di folder 'data/' di dalam bucket
            blob = bucket.blob(f"data/{DATA_FILE}")
            blob.download_to_filename(DATA_FILE)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Kita biarkan lanjut, nanti akan error di pd.read_csv jika file benar2 tidak ada
            
def train_and_evaluate():
    ensure_data_exists()
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL ERROR: File {DATA_FILE} not found!")
        print("Pastikan file 'sentiment-financial.csv' sudah diupload ke folder 'data/' di bucket Anda.")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_FILE, header=None, names=['label', 'text'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # 2. Preprocessing & Splitting
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Pipeline
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
    
    # Format metrics
    metrics = {
        "model_name": "LogisticRegression",
        "parameters": str(model.get_params()),
        "accuracy": acc,
        "f1_scores": {label: score for label, score in zip(labels, f1_scores)}
    }
    
    # 4. Inference Checks
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
    print(f"Saving artifacts...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    with open('metrics.json', 'w') as f:
        json.dump(final_output, f, indent=2)
        
    # Upload artifact ke GCS menggunakan storage client (lebih robust dari gsutil)
    try:
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if bucket_name:
            bucket_name_clean = bucket_name.replace("gs://", "")
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name_clean)
            
            # Upload model & vectorizer
            # Vertex AI mengharapkan model di folder model/ (dari AIP_MODEL_DIR), 
            # tapi kita upload manual saja agar yakin
            
            # Parsing AIP_MODEL_DIR untuk mendapatkan path relatif di bucket
            # AIP_MODEL_DIR format: gs://bucket/path/to/job/output/model
            if MODEL_DIR and MODEL_DIR.startswith("gs://"):
                # Kita gunakan gsutil untuk upload artifact final karena Vertex AI
                # otomatis mount env var ini, tapi pakai python client lebih aman
                pass 
                
            # Fallback pakai gsutil untuk upload output akhir (biasanya sudah available utk write)
            os.system(f"gsutil cp model.joblib {MODEL_DIR}/model.joblib")
            os.system(f"gsutil cp vectorizer.joblib {MODEL_DIR}/vectorizer.joblib")
            os.system(f"gsutil cp metrics.json {MODEL_DIR}/metrics.json")
            print("Artifacts uploaded.")
            
    except Exception as e:
        print(f"Error uploading artifacts: {e}")

    print("Training job finished.")

if __name__ == '__main__':
    train_and_evaluate()
