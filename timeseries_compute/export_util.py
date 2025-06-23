import pandas as pd
import datetime
import inspect
import os
import re
import json
import numpy as np
from typing import Any, Optional, Tuple

# Global step counter and debugging flag
export_data_mode = False  # Default to False, can be changed at runtime

_DATA_STEP_COUNTER = 0
_STATIC_TIMESTAMP = None  # set on first call


def export_data(data: Any, folder: str = "outputs", name: Optional[str] = None) -> Any:
    """
    Save any data to a file with automatically incremented counter and
    inferred variable name. Returns the original data for piping operations.

    Only saves data if export_data_mode is True.

    Args:
        data: The data to save (DataFrame, dict, list, string, etc.)
        folder: Directory to save files in
        name: Optional explicit name to use instead of auto-detection

    Returns:
        The original data (for chaining)
    """
    global _DATA_STEP_COUNTER
    global _STATIC_TIMESTAMP

    # If debugging mode is off, just return the data without saving
    if not export_data_mode:
        return data

    # Generate timestamp only on first call
    if _STATIC_TIMESTAMP is None:
        _STATIC_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _DATA_STEP_COUNTER += 1

    # Create output folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get caller information
    script_name, line_number, variable_name = get_caller_info()

    # Determine file format based on data type
    if isinstance(data, pd.DataFrame):
        file_format = "csv"
    elif isinstance(data, (dict, list)):
        file_format = "json"
    elif isinstance(data, np.ndarray):
        file_format = "npy"
    elif isinstance(data, pd.Series):
        file_format = "json"
    elif hasattr(data, "tolist") and callable(getattr(data, "tolist", None)):
        file_format = "json"
    else:
        file_format = "txt"

    # Construct filename with appropriate extension
    # 20250502_203805--004--example_multivariate_garch#116--var=correlation_matrix.csv
    filename = (
        f"{timestamp}"
        f"--{_DATA_STEP_COUNTER:03d}"
        f"--{script_name}"
        f"#{line_number}"
        f"--var={variable_name}"
        f".{file_format}"
    )

    full_path = os.path.join(folder, filename)

    # Save the data in the appropriate format
    try:
        if file_format == "csv" and isinstance(data, pd.DataFrame):
            data.to_csv(full_path)

        elif file_format == "json":
            with open(full_path, "w") as f:
                if (
                    isinstance(data, (dict, list, int, float, str, bool))
                    or data is None
                ):
                    json.dump(data, f, indent=2, default=str)
                else:
                    # Try to convert to dict or list if possible
                    try:
                        if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")) and not isinstance(data, np.ndarray):
                            json.dump(data.to_dict(), f, indent=2, default=str)
                        elif isinstance(data, pd.Series):
                            # Handle pandas Series specifically
                            json.dump(list(data), f, indent=2, default=str)
                        elif isinstance(data, np.ndarray):
                            # Handle numpy arrays specifically
                            json.dump(data.tolist(), f, indent=2, default=str)
                        elif hasattr(data, "tolist") and callable(getattr(data, "tolist", None)) and not isinstance(data, (pd.Series, pd.DataFrame)):
                            # For other objects with callable tolist method (excluding pandas objects)
                            json.dump(data.tolist(), f, indent=2, default=str)
                        else:
                            json.dump(str(data), f, indent=2)
                    except Exception:
                        json.dump(str(data), f, indent=2)

        elif file_format == "npy" and isinstance(data, np.ndarray):
            np.save(full_path, data)

        else:  # txt or other formats
            with open(full_path, "w") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))

        print(f"Saved: {full_path}")

    except Exception as e:
        print(f"Error saving data: {e}")

    # Return the original data to allow for piping
    return data


def get_caller_info() -> Tuple[str, int, Optional[str]]:
    """
    Extract information about the calling context.

    Returns:
        Tuple of (script_name, line_number, variable_name)
    """
    frame = inspect.currentframe()
    if frame is None:
        return "unknown_script", 0, None

    caller_frame = frame.f_back
    if caller_frame is None:
        return "unknown_script", 0, None

    script_name = os.path.basename(caller_frame.f_code.co_filename).replace(".py", "")
    line_number = caller_frame.f_lineno

    # Try to extract variable name from code context
    variable_name = None
    try:
        frame_info = inspect.getframeinfo(caller_frame)
        if frame_info.code_context:
            context_lines = frame_info.code_context
            if context_lines and len(context_lines) > 0:
                # Look for variable assignment patterns
                line = context_lines[0].strip()
                if "export_data(" in line:
                    # Extract variable name before export_data call
                    if "=" in line and "export_data(" in line:
                        var_part = line.split("export_data(")[0]
                        if "=" in var_part:
                            variable_name = var_part.split("=")[-1].strip()
                        else:
                            variable_name = var_part.strip()
    except Exception:
        # If we can't extract variable name, continue without it
        pass

    return script_name, line_number, variable_name


# Monkey patch DataFrame to add export_data method
# Note: This approach avoids Pylance warnings about attribute assignment
def _export_data_method(self, folder: str = "outputs", name: Optional[str] = None) -> pd.DataFrame:
    """Export DataFrame data using the export_data function."""
    return export_data(self, folder, name)

# Use setattr to avoid Pylance warnings about unknown attributes
setattr(pd.DataFrame, 'export_data', _export_data_method)
