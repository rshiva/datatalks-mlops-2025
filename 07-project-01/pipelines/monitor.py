
import os

# --- Environment Configuration for LocalStack ---
# This MUST be set before the mlflow library is imported.
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:4566"

import pandas as pd
import mlflow
import pickle
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# --- Configuration ---
MODEL_NAME = "MagePersonalityClf"
REFERENCE_DATA_PATH = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/personality_dataset.csv"
CURRENT_DATA_PATH = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/personality_dataset.csv"
REPORT_PATH = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/reports/monitoring_report.html"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))

def monitor_model():
    """
    Loads the latest model, runs it on reference and current data,
    and generates a drift and performance report with Evidently.
    """
    client = mlflow.tracking.MlflowClient()

    # --- 1. Get the Latest Model Version and its Run ID ---
    print(f"Finding latest model: {MODEL_NAME}")
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if not latest_versions:
        raise Exception(f"No models found for name: {MODEL_NAME}")
    latest_version = latest_versions[0]
    run_id = latest_version.run_id
    print(f"Found model version {latest_version.version} from run ID: {run_id}")

    # --- 2. Load Model and Preprocessors using runs:/ URI ---
    # This is the most robust method when the server is correctly configured.
    print("Loading model and preprocessors...")
    model_uri = f"runs:/{run_id}/model"
    preprocessors_uri = f"runs:/{run_id}/preprocessors"

    model = mlflow.pyfunc.load_model(model_uri)
    preprocessors_path = mlflow.artifacts.download_artifacts(artifact_uri=preprocessors_uri)

    with open(os.path.join(preprocessors_path, "scaler.bin"), "rb") as f: scaler = pickle.load(f)
    with open(os.path.join(preprocessors_path, "label_encoder.bin"), "rb") as f: le = pickle.load(f)
    print("Model and preprocessors loaded successfully.")

    # --- 3. Prepare Datasets ---
    print("Preparing reference and current datasets...")
    df_ref = pd.read_csv(REFERENCE_DATA_PATH)
    df_cur = pd.read_csv(CURRENT_DATA_PATH)

    for df in [df_ref, df_cur]:
        features = df[scaler.feature_names_in_]
        features_scaled = scaler.transform(features)
        df['target'] = le.transform(df['personality_type'])
        df['prediction'] = model.predict(features_scaled)

    # --- 4. Generate Evidently Report ---
    print("Generating monitoring report...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])
    report.run(reference_data=df_ref, current_data=df_cur)
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    monitor_model()
