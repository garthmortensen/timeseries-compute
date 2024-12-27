import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

series_1_path = os.path.join(script_dir, "../input/data/series_1.csv")
series_2_path = os.path.join(script_dir, "../input/data/series_2.csv")
series_1 = pd.read_csv(series_1_path)
series_2 = pd.read_csv(series_2_path)
