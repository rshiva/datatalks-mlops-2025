
# Personality Prediction Project

This project aims to predict a person's personality type (Introvert, Extrovert, or Ambivert) based on a series of survey-style questions about their preferences and behaviors.

## Problem Description

The goal of this project is to build a machine learning model that can accurately classify an individual's personality type using their responses to 29 different behavioral and preferential questions. This serves as an end-to-end MLOps demonstration, covering everything from data analysis and experiment tracking to pipeline automation and deployment.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd 07-project-01
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Pipelines

This project includes two main pipelines: one for training the model and one for running batch predictions.

### 1. Training the Model

To train the model and log the experiment to MLflow, run the training script:

```bash
python pipelines/train.py
```

This will:
- Load the `personality_dataset.csv`.
- Preprocess the data.
- Train a Logistic Regression model.
- Log the model, parameters, and metrics to an MLflow experiment named `Personality_Prediction_Pipeline`.

To view the results, run the MLflow UI:
```bash
mlflow ui
```

### 2. Running Batch Predictions

To make predictions on a new dataset using the latest trained model, run the batch prediction script:

```bash
python pipelines/batch_predict.py
```

This will:
- Automatically find the latest model from the `Personality_Prediction_Pipeline` experiment.
- Load the model and its associated preprocessors.
- Make predictions on the data in `personality_dataset.csv`.
- Save the results to `predictions/batch_predictions.csv`.

## Workflow Orchestration with Mage

This project uses Mage for orchestrating the training workflow.

### 1. Start the Mage UI

Navigate to the `mage_project` directory and start the server:

```bash
cd mage_project
python -m mage start .
```

Now open your browser to `http://localhost:6789`.

### 2. Run the Training Pipeline

- In the Mage UI, find the pipeline named `personality_training`.
- Click the "Run pipeline" button to execute the training workflow.
- This will create a new run in the `Mage_Personality_Pipeline` experiment in MLflow.
