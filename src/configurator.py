#!/usr/bin/env python3
# configurator.py

import os
import yaml
import logging as l
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any


# Dict[str, Any] bc the config file's structure is unknown
def read_config_from_fs(config_filename: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file from the config directory.

    Args:
        config_filename (str): The name of the configuration file.

    Returns:
        Dict[str, Any]: The parsed configuration file as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", config_filename)
    with open(config_path, "r") as f:
        try:
            contents = yaml.safe_load(f)
            l.info(f"yml contents:\n{contents}")
        except ValidationError as e:  # if the yml is not valid
            l.info(f"Validation error: {e}")
        return contents


class DataGeneratorConfig(BaseModel):
    enabled: bool = Field(default=True)  # New field to capture "enabled" parameter
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2023-12-31")
    anchor_prices: Dict[str, float] = Field(default_factory=dict)


class MissingDataHandlerConfig(BaseModel):
    enabled: bool = Field(default=True)
    strategy: str = Field(default="forward_fill")


class ScalingConfig(BaseModel):
    enabled: bool = Field(default=True)
    method: str = Field(default="standardize")


class MakeStationarityConfig(BaseModel):
    enabled: bool = Field(default=True)
    method: str = Field(default="difference")


class TestStationarityConfig(BaseModel):
    method: str = Field(default="ADF")
    p_value_threshold: float = Field(default=0.05)


class DataProcessorConfig(BaseModel):
    handle_missing_values: MissingDataHandlerConfig = Field(
        default_factory=MissingDataHandlerConfig
    )
    make_stationary: MakeStationarityConfig = Field(
        default_factory=MakeStationarityConfig
    )
    test_stationarity: TestStationarityConfig = Field(
        default_factory=TestStationarityConfig
    )
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)


class ARIMAConfig(BaseModel):
    # lambda used to set default dictionary values
    enabled: bool = Field(default=False)
    parameters_fit: Dict[str, int] = Field(
        default_factory=lambda: {"p": 1, "d": 1, "q": 1}
    )
    parameters_predict_steps: int = Field(default=5)


# because this section is nested in the json, we need to define a separate class for it
class GARCHParametersFitConfig(BaseModel):
    p: int = Field(default=1)
    q: int = Field(default=1)
    dist: str = Field(default="normal")


class GARCHConfig(BaseModel):
    enabled: bool = Field(default=False)
    parameters_fit: GARCHParametersFitConfig = Field(
        default_factory=GARCHParametersFitConfig
    )
    parameters_predict_steps: int = Field(default=5)


class StatsModelConfig(BaseModel):
    ARIMA: ARIMAConfig = Field(default_factory=ARIMAConfig)
    GARCH: GARCHConfig = Field(default_factory=GARCHConfig)


class Config(BaseModel):
    """
    This class defines the configuration file structure.
    It's instantiated and returned by the load_configuration function.
    `data_generator` becomes part of the drill down to access the start_date, end_date, and anchor_prices.
    `default_factory` is pydantic field simpy for specifying default values.
    """

    data_generator: DataGeneratorConfig = Field(default_factory=DataGeneratorConfig)
    data_processor: DataProcessorConfig = Field(default_factory=DataProcessorConfig)
    stats_model: StatsModelConfig = Field(default_factory=StatsModelConfig)


def load_configuration(config_file: str) -> Config:
    """
    Load and validate the YAML configuration file.

    Args:
        config_file (str): The name of the configuration file.

    Returns:
        Config: The validated configuration object.
    """
    l.info(f"# Loading config_file: {config_file}")
    config_dict: Dict[str, Any] = read_config_from_fs(config_file)
    return Config(**config_dict)  # ** unpacks the dictionary into keyword args
