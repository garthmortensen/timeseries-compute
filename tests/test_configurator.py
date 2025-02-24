# tests/test_configurator.py

import pytest
import yaml
from src.configurator import read_config_from_fs, MissingDataHandlerConfig, ScalingConfig

sample_config = {
    "data_generator": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "anchor_prices": {"GME": 150.5, "BYND": 700.0}
    },
    "data_processor": {
        "handle_missing_values": {"strategy": "drop"},
        "scaling": {"method": "standardize"}
    }
}

@pytest.fixture
# fixture that returns a path to a temp file
def config_file(tmp_path):
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path

def test_read_config_from_fs(config_file):
    config = read_config_from_fs(config_file)
    assert config["data_generator"]["start_date"] == "2023-01-01"
    assert config["data_generator"]["end_date"] == "2023-12-31"
    assert config["data_generator"]["anchor_prices"]["GME"] == 150.5

def test_missing_data_handler_config():
    config = MissingDataHandlerConfig(strategy="forward_fill")
    assert config.strategy == "forward_fill"

def test_scaling_config():
    config = ScalingConfig(method="standardize")
    assert config.method == "standardize"

