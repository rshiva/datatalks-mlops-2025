if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os

@transformer
def preprocess_and_train(df, *args, **kwargs):
    """
    Preprocesses the data, trains a model, and logs to MLflow.
    """
    # Set up MLflow
    mlflow.set_experiment("Mage_Personality_Pipeline")

    with mlflow.start_run(run_name="Mage_Training_Run"):
        # --- Preprocessing ---
        le = LabelEncoder()
        df['personality_type_encoded'] = le.fit_transform(df['personality_type'])
        X = df.drop(['personality_type', 'personality_type_encoded'], axis=1)
        y = df['personality_type_encoded']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data preprocessing complete.")

        # --- Training ---
        params = {"random_state": 42, "max_iter": 1000}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        print("Model training complete.")
        # --- Evaluation & Logging ---
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        print(f"Validation Accuracy: {accuracy:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="MagePersonalityClf"
        )

        # --- Return trained model and preprocessors ---
        return {
            "model": model,
            "scaler": scaler,
            "label_encoder": le
        }

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
