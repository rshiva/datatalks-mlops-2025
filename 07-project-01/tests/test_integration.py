
import os
import pandas as pd
from pipelines.batch_predict import predict_from_registry

# This is a simplified integration test.
# A more robust test would involve a dedicated test model on the registry.

def test_batch_prediction_pipeline():
    """ 
    Tests the batch prediction pipeline by running it on a small sample
    of data and checking the output.
    """
    # --- Arrange ---
    # Create a dummy input file
    sample_data = {
        'social_energy': [6.0], 'alone_time_preference': [4.0], 'talkativeness': [8.0],
        'deep_reflection': [2.0], 'group_comfort': [7.0], 'party_liking': [8.0],
        'listening_skill': [6.0], 'empathy': [6.0], 'creativity': [6.0],
        'organization': [0.0], 'leadership': [8.0], 'risk_taking': [7.0],
        'public_speaking_comfort': [8.0], 'curiosity': [5.0], 'routine_preference': [4.0],
        'excitement_seeking': [8.0], 'friendliness': [8.0], 'emotional_stability': [5.0],
        'planning': [4.0], 'spontaneity': [5.0], 'adventurousness': [8.0],
        'reading_habit': [5.0], 'sports_interest': [10.0], 'online_social_usage': [9.0],
        'travel_desire': [5.0], 'gadget_usage': [9.0], 'work_style_collaborative': [8.0],
        'decision_speed': [8.0], 'stress_handling': [7.0]
    }
    input_df = pd.DataFrame(sample_data)
    input_path = "/tmp/test_input.csv"
    output_path = "/tmp/test_output.csv"
    input_df.to_csv(input_path, index=False)

    # --- Act ---
    # Run the prediction pipeline
    # NOTE: This requires a model to be in the "Staging" phase in the registry.
    try:
        predict_from_registry("PersonalityClf", "Staging", input_path, output_path)
    except Exception as e:
        assert False, f"Prediction pipeline failed with an exception: {e}"

    # --- Assert ---
    assert os.path.exists(output_path), "Output file was not created."
    
    result_df = pd.read_csv(output_path)
    assert "predicted_personality" in result_df.columns, "Prediction column is missing."
    assert len(result_df) == 1, "Output file should have one prediction."

    # --- Cleanup ---
    os.remove(input_path)
    os.remove(output_path)
