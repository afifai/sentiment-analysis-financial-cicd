import os
import json
import argparse
import sys
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage
# Import client level rendah untuk Import Evaluation
from google.cloud import aiplatform_v1

def submit_custom_job(
    project_id, 
    bucket_name, 
    branch_name,
    script_path="src/train.py"
):
    location = "us-central1"
    staging_bucket = bucket_name 
    
    # Init SDK High Level
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
    model_artifacts_dir = JOB_OUTPUT_DIR + "/model"
    print(f"✅ Job finished. Expected Artifacts: {model_artifacts_dir}")
    
    storage_client = storage.Client(project=project_id)
    bucket_name_clean = bucket_name.replace("gs://", "")
    gcs_bucket = storage_client.bucket(bucket_name_clean)
    
    # Logic Path metrics.json
    blob_prefix = model_artifacts_dir.replace(f"gs://{bucket_name_clean}/", "")
    if blob_prefix.startswith("/"):
        blob_prefix = blob_prefix[1:]
        
    blob_path = f"{blob_prefix}/metrics.json"
    
    print(f"🔎 Looking for file in Bucket: {bucket_name_clean}")
    print(f"🔎 Full Blob Path: {blob_path}")
    
    blob = gcs_bucket.blob(blob_path)
    
    if not blob.exists():
        print("❌ CRITICAL: metrics.json NOT FOUND in GCS!")
        print("Please check previous logs if 'src/train.py' uploaded it successfully.")
        sys.exit(1)
    
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
    
    # --- Import Evaluation (FIXED VERSION) ---
    print("📊 Importing Evaluation metrics...")
    
    try:
        # Prepare metrics dictionary
        eval_metrics = {
            "accuracy": metrics['accuracy'],
        }
        for k, v in metrics['f1_scores'].items():
            eval_metrics[f"f1_score_{k}"] = v
            
        # Gunakan Client API v1 (GAPIC)
        # Kita tidak pakai metrics_schema_uri agar bisa menerima custom keys
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform_v1.ModelServiceClient(client_options=client_options)
        
        model_eval_payload = {
            "display_name": f"eval-{job_id_unique}",
            "metrics": eval_metrics,
            "metadata": {
                "pipeline_job_resource_name": job.resource_name
            }
        }
        
        request = aiplatform_v1.ImportModelEvaluationRequest(
            parent=model.resource_name,
            model_evaluation=model_eval_payload
        )
        
        client.import_model_evaluation(request=request)
        print(f"✅ Evaluation imported successfully to {model.resource_name}")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to import evaluation metrics (non-fatal): {e}")
        # Kita lanjut saja agar pipeline tidak merah total hanya gara-gara metrics display
        
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
