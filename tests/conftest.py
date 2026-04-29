import pytest
import pandas as pd
import os
import sys

# Add src to path so pipeline imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add dashboard to path so StyleFeatureExtractorV6 can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dashboard'))


@pytest.fixture
def sample_texts():
    """A mix of reliable and suspect texts in EN and FR."""
    return pd.Series([
        'New study published in Nature confirms the effectiveness of the updated vaccine formula.',
        'EXPOSED: Secret government labs are using 5G towers to spread mind-control chemicals!!!',
        'SCANDALE: le gouvernement cache la VERITE sur les chemtrails! Partagez avant censure!!!',
        'The weather is nice today.',
        'Breaking news from Reuters: the economy grew by 2% this quarter.',
    ])


@pytest.fixture
def model_dir():
    """Path to the models directory."""
    return os.path.join(os.path.dirname(__file__), '..', 'models')
