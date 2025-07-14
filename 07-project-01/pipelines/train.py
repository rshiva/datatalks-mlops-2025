import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os

# 1. Set up MLflow Experiment
mlflow.set_experiment("Personality_Prediction_Pipeline")


def train_model(data_path: str):
    """
    Loads data, preprocesses, trains a Logistic Regression model,
    and logs everything to MLflow.
    """
    with mlflow.start_run(run_name="Training_Pipeline_Run"):
        # 2. Load and Preprocess Data
        print("Loading and preprocessing data...")
        df = pd.read_csv(data_path)

        # Label Encoding for the target variable
        le = LabelEncoder()
        df["personality_type_encoded"] = le.fit_transform(df["personality_type"])

        # Separate features (X) and target (y)
        X = df.drop(["personality_type", "personality_type_encoded"], axis=1)
        y = df["personality_type_encoded"]

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data preprocessing complete.")

        #  3. Train Model
        print("Training Logistic Regression model...")
        params = {"random_state": 42, "max_iter": 1000}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        print("Model training complete.")

        #  4. Evaluate and Log Metrics
        print("Evaluating model...")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1-Score: {f1:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("model_type", "LogisticRegression")

        #  5. Log Model and Artifacts
        print("Logging model and artifacts to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_regression_model",
            registered_model_name="PersonalityClf",
            # registered_model_aliases=["staging"]  # Set the alias here
        )

        # Save the scaler and label encoder to a local directory
        preprocessors_dir = "preprocessors"
        os.makedirs(preprocessors_dir, exist_ok=True)
        
        with open(os.path.join(preprocessors_dir, "scaler.bin"), "wb") as f_out:
            pickle.dump(scaler, f_out)
        with open(os.path.join(preprocessors_dir, "label_encoder.bin"), "wb") as f_out:
            pickle.dump(le, f_out)

        # Log the entire directory as an artifact
        mlflow.log_artifacts(preprocessors_dir, artifact_path="preprocessors")

        print("Training pipeline finished successfully!")


if __name__ == "__main__":
    # The path to your dataset
    DATA_PATH = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/personality_dataset.csv"
    train_model(DATA_PATH)
