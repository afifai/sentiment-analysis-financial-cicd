import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Try import native storage client
try:
    from google.cloud import storage
    HAS_STORAGE_LIB = True
except ImportError:
    HAS_STORAGE_LIB = False

# --- KONFIGURASI ENV VERTEX AI ---
MODEL_DIR = os.getenv('AIP_MODEL_DIR', '.')
DATA_FILE = 'sentiment-financial.csv'

def download_from_gcs_robust(bucket_url, source_path, dest_path):
    print(f"Attempting to download {source_path} from {bucket_url}...")
    
    # Method 1: Python Library
    if HAS_STORAGE_LIB:
        try:
            print("Method 1: Using google-cloud-storage library...")
            bucket_name_clean = bucket_url.replace("gs://", "")
            client = storage.Client()
            bucket = client.bucket(bucket_name_clean)
            blob = bucket.blob(source_path)
            blob.download_to_filename(dest_path)
            print("Download successful (Method 1).")
            return True
        except Exception as e:
            print(f"Method 1 failed: {e}")
    
    # Method 2: gsutil CLI
    print("Method 2: Using gsutil CLI...")
    try:
        full_source_uri = f"{bucket_url}/{source_path}"
        cmd = f"gsutil cp {full_source_uri} {dest_path}"
        subprocess.check_call(cmd, shell=True)
        print("Download successful (Method 2).")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Method 2 failed: {e}")
        return False

def train_and_evaluate():
    # 1. Download Data Logic
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if bucket_name:
        success = download_from_gcs_robust(bucket_name, f"data/{DATA_FILE}", DATA_FILE)
        if not success:
            print("Warning: Failed to download data from GCS. Checking local file...")
    else:
        print("Warning: GCS_BUCKET_NAME not set. Using local file if exists.")

    # 2. Load Data
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL ERROR: File {DATA_FILE} not found!")
        sys.exit(1)

    try:
        # FIXED: Menambahkan encoding='latin-1' untuk handle karakter spesial
        df = pd.read_csv(DATA_FILE, header=None, names=['label', 'text'], encoding='latin-1')
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # 3. Preprocessing
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Pipeline
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    # 4. Evaluasi
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    labels = model.classes_
    f1_scores = f1_score(y_test, y_pred, average=None, labels=labels)
    
    metrics = {
        "model_name": "LogisticRegression",
        "parameters": str(model.get_params()),
        "accuracy": acc,
        "f1_scores": {label: score for label, score in zip(labels, f1_scores)}
    }
    
    # 5. Inference Checks
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

    # 6. Save Artifacts
    print(f"Saving artifacts...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    with open('metrics.json', 'w') as f:
        json.dump(final_output, f, indent=2)
        
    # Upload Artifacts
    print("Uploading artifacts to GCS...")
    try:
        os.system(f"gsutil cp model.joblib {MODEL_DIR}/model.joblib")
        os.system(f"gsutil cp vectorizer.joblib {MODEL_DIR}/vectorizer.joblib")
        os.system(f"gsutil cp metrics.json {MODEL_DIR}/metrics.json")
    except Exception as e:
        print(f"Error uploading artifacts: {e}")
    
    print("Training job finished successfully.")

if __name__ == '__main__':
    train_and_evaluate()
