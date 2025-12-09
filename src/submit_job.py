import os
import json
import argparse
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
        # Jika main, kasih nama spesial
        identifier = "selected-jobs"
    else:
        # Jika features/kelompok-1 -> ambil 'kelompok-1'
        # Jika feat/kelompok-1 -> ambil 'kelompok-1'
        # Jika kelompok-1 (tanpa slash) -> ambil 'kelompok-1'
        if "/" in branch_name:
            identifier = branch_name.split("/")[-1]
        else:
            identifier = branch_name
    
    # Bersihkan identifier dari karakter aneh agar aman buat nama job
    identifier = identifier.replace("_", "-").lower()
    
    job_id_unique = f"{identifier}-{timestamp}"
    display_name = f"train-{job_id_unique}"
    
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
    print(f"📦 Container: {container_uri}")
    
    job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path=script_path,
        container_uri=container_uri,
        requirements=requirements,
        replica_count=1,
        machine_type="n1-standard-4",
        environment_variables={"GCS_BUCKET_NAME": bucket_name} 
    )
    
    job.run(sync=True)
    
    # --- Retrieving Artifacts ---
    model_artifacts_dir = job.base_output_dir + "/model"
    print(f"✅ Job finished. Artifacts: {model_artifacts_dir}")
    
    storage_client = storage.Client(project=project_id)
    bucket_name_clean = bucket_name.replace("gs://", "")
    gcs_bucket = storage_client.bucket(bucket_name_clean)
    
    # Download metrics.json
    blob_path = model_artifacts_dir.replace(f"gs://{bucket_name_clean}/", "") + "/metrics.json"
    blob = gcs_bucket.blob(blob_path)
    
    print(f"📥 Downloading metrics from: {blob_path}")
    try:
        blob.download_to_filename("metrics.json")
    except Exception as e:
        print(f"❌ Error downloading metrics: {e}")
        return
    
    with open("metrics.json", "r") as f:
        data = json.load(f)
    metrics = data['metrics']
    
    # --- Register Model ---
    print("📤 Uploading model to Registry...")
    
    serving_image = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
    
    model = aiplatform.Model.upload(
        display_name=f"model-{identifier}", # Nama model di registry ikut nama identifier
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
