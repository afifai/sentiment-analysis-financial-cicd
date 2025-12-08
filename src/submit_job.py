import os
import json
import argparse
from google.cloud import aiplatform
from google.cloud import storage

def submit_custom_job(
    project_id, 
    bucket_name, 
    branch_name,
    script_path="src/train.py"
):
    location = "us-central1" # Sesuaikan region
    staging_bucket = bucket_name 
    
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)
    
    is_prod = (branch_name == "main")
    display_name = f"train-{branch_name.replace('/', '-')}"
    
    # 1. Submit Training Job
    job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path=script_path,
        container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest",
        requirements=["pandas", "scikit-learn", "numpy", "joblib", "google-cloud-storage"],
        replica_count=1,
        machine_type="n1-standard-4",
        environment_variables={"GCS_BUCKET_NAME": bucket_name} 
    )
    
    print(f"Submitting Job: {display_name}")
    job.run(sync=True)
    
    # 2. Retrieve Metrics Artifacts
    model_artifacts_dir = job.base_output_dir + "/model"
    print(f"Job finished. Artifacts: {model_artifacts_dir}")
    
    storage_client = storage.Client(project=project_id)
    gcs_bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    # Parse path GCS untuk metrics.json
    # Format: .../model/metrics.json
    blob_path = model_artifacts_dir.replace(f"{bucket_name}/", "") + "/metrics.json"
    
    blob = gcs_bucket.blob(blob_path)
    blob.download_to_filename("metrics.json")
    
    with open("metrics.json", "r") as f:
        data = json.load(f)
    metrics = data['metrics']
    
    # 3. Upload & Register Model ke Vertex AI Registry
    print("Uploading model to Registry...")
    model = aiplatform.Model.upload(
        display_name="financial-sentiment-model",
        artifact_uri=model_artifacts_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest",
        is_default_version=True
    )
    
    # 4. Import Evaluation Metrics (Agar tab Evaluate muncul)
    eval_metrics = {
        "accuracy": metrics['accuracy'],
    }
    # Flatten F1 scores
    for k, v in metrics['f1_scores'].items():
        eval_metrics[f"f1_score_{k}"] = v

    print("Importing Evaluation metrics...")
    model.import_evaluation(
        display_name=f"eval-{branch_name}",
        metrics=eval_metrics,
        pipeline_job_id=job.name
    )
    
    print(f"Model {model.resource_name} updated.")

    # 5. Update Production Baseline (Jika Main)
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