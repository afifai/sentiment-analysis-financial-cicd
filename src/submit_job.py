import os
import json
import argparse
import sys
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage

def submit_custom_job(
    project_id, 
    bucket_name, 
    branch_name,
    script_path="src/train.py"
):
    location = "us-central1"
    staging_bucket = bucket_name 
    
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)
    
    is_prod = (branch_name == "main")
    
    # --- LOGIC PENAMAAN OTOMATIS ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if branch_name == "main":
        identifier = "selected-jobs"
    else:
        if "/" in branch_name:
            identifier = branch_name.split("/")[-1]
        else:
            identifier = branch_name
    
    identifier = identifier.replace("_", "-").lower()
    
    job_id_unique = f"{identifier}-{timestamp}"
    display_name = f"train-{job_id_unique}"
    
    # --- OUTPUT DIR SETUP ---
    # Hilangkan trailing slash jika ada
    if bucket_name.endswith("/"):
        bucket_name = bucket_name[:-1]
        
    JOB_OUTPUT_DIR = f"{bucket_name}/{display_name}"
    
    # --- CONTAINER SETUP ---
    container_uri = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-14.py310:latest"
    
    requirements = [
        "scikit-learn", 
        "pandas", 
        "joblib", 
        "google-cloud-storage",
        "python-json-logger"
    ]
    
    print(f"🚀 Submitting Job: {display_name}")
    print(f"📂 Output Dir: {JOB_OUTPUT_DIR}")
    
    job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path=script_path,
        container_uri=container_uri,
        requirements=requirements,
        replica_count=1,
        machine_type="n1-standard-4",
        base_output_dir=JOB_OUTPUT_DIR,
        environment_variables={"GCS_BUCKET_NAME": bucket_name} 
    )
    
    job.run(sync=True)
    
    # --- Retrieving Artifacts ---
    # Vertex AI otomatis membuat folder 'model' di dalam output dir
    model_artifacts_dir = JOB_OUTPUT_DIR + "/model"
    print(f"✅ Job finished. Expected Artifacts: {model_artifacts_dir}")
    
    storage_client = storage.Client(project=project_id)
    bucket_name_clean = bucket_name.replace("gs://", "")
    gcs_bucket = storage_client.bucket(bucket_name_clean)
    
    # Logic Path metrics.json
    # Contoh: gs://bucket/train-id/model/metrics.json
    # Blob path: train-id/model/metrics.json
    
    # Hapus awalan bucket dari path artifact
    blob_prefix = model_artifacts_dir.replace(f"gs://{bucket_name_clean}/", "")
    # Pastikan tidak ada double slash
    if blob_prefix.startswith("/"):
        blob_prefix = blob_prefix[1:]
        
    blob_path = f"{blob_prefix}/metrics.json"
    
    print(f"🔎 Looking for file in Bucket: {bucket_name_clean}")
    print(f"🔎 Full Blob Path: {blob_path}")
    
    blob = gcs_bucket.blob(blob_path)
    
    if not blob.exists():
        print("❌ CRITICAL: metrics.json NOT FOUND in GCS!")
        print("Possible causes:")
        print("1. Training script failed before saving metrics.")
        print("2. Upload logic in train.py failed.")
        print("3. Path mismatch.")
        sys.exit(1) # Paksa Fail disini agar tidak lanjut ke compare_metrics
    
    print(f"📥 Downloading metrics...")
    blob.download_to_filename("metrics.json")
    print("✅ Download success.")
    
    with open("metrics.json", "r") as f:
        data = json.load(f)
    metrics = data['metrics']
    
    # --- Register Model ---
    print("📤 Uploading model to Registry...")
    
    serving_image = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
    
    model = aiplatform.Model.upload(
        display_name=f"model-{identifier}", 
        artifact_uri=model_artifacts_dir,
        serving_container_image_uri=serving_image,
        is_default_version=True,
        labels={"source": "github-actions", "branch": identifier}
    )
    
    # --- Import Evaluation ---
    eval_metrics = {
        "accuracy": metrics['accuracy'],
    }
    for k, v in metrics['f1_scores'].items():
        eval_metrics[f"f1_score_{k}"] = v

    print("📊 Importing Evaluation metrics...")
    model.import_evaluation(
        display_name=f"eval-{job_id_unique}",
        metrics=eval_metrics,
        pipeline_job_id=job.name
    )
    
    print(f"✅ Model Resource Name: {model.resource_name}")

    if is_prod:
        baseline_blob = gcs_bucket.blob("prod_baseline/metrics.json")
        baseline_blob.upload_from_filename("metrics.json")
        print("Updated Production Baseline Metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--branch", type=str, required=True)
    args = parser.parse_args()
    
    submit_custom_job(args.project_id, args.bucket, args.branch)
