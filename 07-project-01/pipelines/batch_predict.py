import pandas as pd
import mlflow
import pickle
import os

from mlflow.tracking import MlflowClient

# Set tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Configuration ---
MODEL_NAME = "PersonalityClf"

def predict_from_registry(model_name: str, data_path: str, output_path: str):
    """
    Loads the latest model version from MLflow Model Registry,
    makes predictions, and saves them to CSV.
    """
    client = MlflowClient()

    # --- 1. Get Latest Version of the Model ---
    latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
    if not latest_version:
        raise Exception(f"No versions found for model '{model_name}'")

    # Get the most recent version (pick first for now)
    model_version_info = latest_version[0]
    run_id = model_version_info.run_id
    version_number = model_version_info.version
    model_uri = f"models:/{model_name}/{version_number}"

    print(f"Using model version: {version_number}, Run ID: {run_id}, model_uri: {model_uri}")

    # --- 2. Load the Model from MLflow ---
    model = mlflow.pyfunc.load_model(model_uri)

    # --- 3. Load Preprocessors (Scaler + LabelEncoder) from Artifacts ---
    print("Downloading and loading preprocessors...")
    client.download_artifacts(run_id, "preprocessors", ".")

    with open("preprocessors/scaler.bin", "rb") as f:
        scaler = pickle.load(f)

    with open("preprocessors/label_encoder.bin", "rb") as f:
        le = pickle.load(f)

    # --- 4. Load and Preprocess Input Data ---
    print(f"Loading input data from: {data_path}")
    df_new = pd.read_csv(data_path)

    # Ensure we select the same features used in training
    X_new = df_new[scaler.feature_names_in_]
    X_new_scaled = scaler.transform(X_new)

    # --- 5. Run Predictions ---
    print("Running predictions...")
    preds_encoded = model.predict(X_new_scaled)
    preds = le.inverse_transform(preds_encoded)

    # --- 6. Save Results ---
    df_results = pd.DataFrame({
        "original_data": df_new.to_dict(orient="records"),
        "predicted_personality": preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

# --- Entry point ---
if __name__ == "__main__":
    INPUT_DATA_PATH = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/personality_dataset.csv"
    OUTPUT_DATA_PATH = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/predictions/batch_predictions.csv"

    predict_from_registry(MODEL_NAME, INPUT_DATA_PATH, OUTPUT_DATA_PATH)
