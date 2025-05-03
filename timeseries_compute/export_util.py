import pandas as pd
import datetime
import inspect
import os
import re
import json
import numpy as np
from typing import Any, Optional

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
    frame = inspect.currentframe().f_back
    script_name = os.path.basename(frame.f_code.co_filename).replace('.py', '')
    line_number = frame.f_lineno
    
    # Try to determine variable name from context if not provided
    variable_name = name if name is not None else "unnamed_data"
    if name is None:
        try:
            context_lines = inspect.getframeinfo(frame).code_context
            if context_lines:
                line = context_lines[0].strip()
                
                # Look for assignment patterns
                match = re.match(r'(\w+)\s*=', line)
                if match:
                    variable_name = match.group(1)
                else:
                    # Look for function call patterns
                    match = re.search(r'export_data\((\w+)', line)
                    if match:
                        variable_name = match.group(1)
        except Exception:
            pass
    
    # Determine file format based on data type
    if isinstance(data, pd.DataFrame):
        file_format = 'csv'
    elif isinstance(data, (dict, list)) or (hasattr(data, 'tolist') and callable(data.tolist)):
        file_format = 'json'
    elif isinstance(data, np.ndarray):
        file_format = 'npy'
    else:
        file_format = 'txt'
    
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
        if file_format == 'csv':
            data.to_csv(full_path)
        
        elif file_format == 'json':
            with open(full_path, 'w') as f:
                if isinstance(data, (dict, list, int, float, str, bool)) or data is None:
                    json.dump(data, f, indent=2, default=str)
                else:
                    # Try to convert to dict or list if possible
                    try:
                        if hasattr(data, 'to_dict'):
                            json.dump(data.to_dict(), f, indent=2, default=str)
                        elif hasattr(data, 'tolist') and callable(data.tolist):
                            json.dump(data.tolist(), f, indent=2, default=str)
                        else:
                            json.dump(str(data), f, indent=2)
                    except:
                        json.dump(str(data), f, indent=2)
        
        elif file_format == 'npy':
            np.save(full_path, data)
        
        else:  # txt or other formats
            with open(full_path, 'w') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))
        
        print(f"Saved: {full_path}")
        
    except Exception as e:
        print(f"Error saving data: {e}")
    
    # Return the original data to allow for piping
    return data


# Only add the method to DataFrame, which is mutable
pd.DataFrame.export_data = lambda self, folder="outputs", name=None: export_data(self, folder, name)
