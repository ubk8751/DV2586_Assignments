import pandas as pd

data_path = ""

df_small_noise = pd.read_csv(
    data_path, parse_dates=True, index_col="timestamp"
)