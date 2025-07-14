
import pandas as pd
import pytest

# Imagine we have a function like this in a `utils.py` file
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """A hypothetical data cleaning function."""
    # Example: Ensure no negative values in a specific column
    if 'risk_taking' in df.columns:
        df['risk_taking'] = df['risk_taking'].clip(lower=0)
    return df

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        'risk_taking': [5, -2, 10, 0],
        'talkativeness': [8, 3, 7, 5]
    })

def test_clean_data_handles_negatives(sample_dataframe):
    """Tests that the clean_data function correctly clips negative values."""
    # Given
    df_dirty = sample_dataframe

    # When
    df_clean = clean_data(df_dirty)

    # Then
    assert (df_clean['risk_taking'] >= 0).all()
    assert df_clean['risk_taking'].iloc[1] == 0
