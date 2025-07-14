if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


import pandas as pd
@data_loader
def load_data(*args, **kwargs):
    """
    Loads the personality dataset from a local CSV file.
    """
    data_path = "/Users/shiva/workspace/mlops/datatalks-mlops-2025/07-project-01/personality_dataset.csv"
    df = pd.read_csv(data_path)
    print("Data loaded successfully with shape:", df.shape)
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
