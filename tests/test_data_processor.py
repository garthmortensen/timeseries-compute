# tests/test_data_processor.py

import pytest
import pandas as pd
from generalized_timeseries.data_processor import MissingDataHandler, MissingDataHandlerFactory

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "A": [1, 2, None, 4, 5],
        "B": [None, 2, 3, None, 5],
        "C": [1, None, None, 4, 5]
    }
    return pd.DataFrame(data)

def test_create_handler_drop():
    """Test the create_handler method of MissingDataHandlerFactory with 'drop' strategy."""
    handler_func = MissingDataHandlerFactory.create_handler("drop")
    assert callable(handler_func)
    assert handler_func.__name__ == "drop_na"

def test_create_handler_forward_fill():
    """Test the create_handler method of MissingDataHandlerFactory with 'forward_fill' strategy."""
    handler_func = MissingDataHandlerFactory.create_handler("forward_fill")
    assert callable(handler_func)
    assert handler_func.__name__ == "forward_fill"

def test_create_handler_invalid_strategy():
    """Test the create_handler method of MissingDataHandlerFactory with an invalid strategy."""
    with pytest.raises(ValueError):
        MissingDataHandlerFactory.create_handler("invalid_strategy")
