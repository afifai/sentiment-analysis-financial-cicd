import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import subprocess
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Import Google Cloud Storage

try:
    from google.cloud import storage
    HAS_STORAGE_LIB = True
except ImportError:
    HAS_STORAGE_LIB = False
    print("Warning: google-cloud-storage library not found.")

# --- KONFIGURASI ENV VERTEX AI ---
# Vertex AI otomatis set AIP_MODEL_DIR ke gs://bucket/output_dir/model
MODEL_DIR = os.getenv('AIP_MODEL_DIR', '.')
DATA_FILE = 'sentiment-financial.csv'

def get_bucket_and_blob_name(gs_path):
    """Helper untuk memecah path gs://bucket/folder/file"""
    if not gs_path.startswith("gs://"):
        return None, None
    parts = gs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name

def upload_to_gcs(local_path, gs_path):
    """Upload file ke GCS menggunakan Storage Client (Lebih stabil dari gsutil)"""
    print(f"Uploading {local_path} to {gs_path}...")
    
    bucket_name, blob_name_prefix = get_bucket_and_blob_name(gs_path)
    
    if not bucket_name:
        print(f"Error: Invalid GCS path {gs_path}")
        return False

    # Jika gs_path adalah folder (diakhiri / atau tidak ada ekstensi), tambahkan nama file lokal
    if blob_name_prefix and not blob_name_prefix.endswith(os.path.basename(local_path)):
         # Pastikan formatnya folder/filename.ext
         if blob_name_prefix.endswith("/"):
             blob_name = blob_name_prefix + os.path.basename(local_path)
         else:
             blob_name = blob_name_prefix + "/" + os.path.basename(local_path)
    else:
        blob_name = blob_name_prefix

    # Clean double slashes just in case
    blob_name = blob_name.replace("//", "/")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"✅ Successfully uploaded to gs://{bucket_name}/{blob_name}")
        return True
    except Exception as e:
        print(f"❌ Upload failed using Python Client: {e}")
        return False

def download_from_gcs_robust(bucket_url, source_path, dest_path):
    print(f"Attempting to download {source_path} from {bucket_url}...")
    
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
    
    # Method 2: gsutil CLI Fallback
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
    # 1. Download Data
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if bucket_name:
        success = download_from_gcs_robust(bucket_name, f"data/{DATA_FILE}", DATA_FILE)
        if not success:
            print("Warning: Failed to download data from GCS.")
            
    # 2. Load Data
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL ERROR: File {DATA_FILE} not found!")
        sys.exit(1)

    try:
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
    vectorizer = CountVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    # model.predict_proba(X_test[:1])
    # model = LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.5)
    # model.fit(X_train_vec, y_train)
    
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

    # 6. Save Artifacts Local
    print(f"Saving artifacts locally...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    with open('metrics.json', 'w') as f:
        json.dump(final_output, f, indent=2)
        
    # 7. Upload to GCS (CRITICAL STEP)
    print(f"Uploading artifacts to MODEL_DIR: {MODEL_DIR}")
    
    if MODEL_DIR and MODEL_DIR.startswith("gs://"):
        # Upload menggunakan Python Client (Preferred)
        upload_to_gcs('model.joblib', MODEL_DIR)
        upload_to_gcs('vectorizer.joblib', MODEL_DIR)
        upload_to_gcs('metrics.json', MODEL_DIR)
    else:
        # Fallback local copy (jarang terjadi di Vertex)
        print(f"MODEL_DIR is local or empty: {MODEL_DIR}. Skipping GCS upload.")
    
    print("Training job finished successfully.")

if __name__ == '__main__':
    train_and_evaluate()
