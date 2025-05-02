import pandas as pd
import datetime
import inspect
import os
import re
from typing import Optional

# Global step counter
_DF_STEP_COUNTER = 0

def export_df(df: pd.DataFrame, folder: str = "outputs") -> pd.DataFrame:
    """
    Save a DataFrame to CSV with automatically incremented counter and 
    inferred variable name. Returns the same DataFrame for piping operations.
    
    Args:
        df: The DataFrame to save
        folder: Directory to save files in
    
    Returns:
        The original DataFrame (for chaining)
    """
    global _DF_STEP_COUNTER
    _DF_STEP_COUNTER += 1
    
    # Create output folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get caller information
    frame = inspect.currentframe().f_back
    script_name = os.path.basename(frame.f_code.co_filename).replace('.py', '')
    line_number = frame.f_lineno
    
    # Try to determine variable name from context
    variable_name = "unnamed_df"
    try:
        context_lines = inspect.getframeinfo(frame).code_context
        if context_lines:
            line = context_lines[0].strip()
            
            # Look for assignment patterns like "variable_name = ...save_df"
            # This handles cases where save_df is used at the end of a line
            match = re.match(r'(\w+)\s*=', line)
            if match:
                variable_name = match.group(1)
    except Exception:
        pass
    
    # Construct filename
    filename = f"{_DF_STEP_COUNTER:03d}_{timestamp}_{script_name}_{line_number}_{variable_name}.csv"
    full_path = os.path.join(folder, filename)
    
    # Save the DataFrame
    df.to_csv(full_path)
    print(f"Saved: {full_path}")
    
    # Return the DataFrame to allow for piping
    return df
