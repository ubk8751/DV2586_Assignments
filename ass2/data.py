import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


def _get_df(data_path:str = "./ass2/ass2data.csv"):
    df = pd.read_csv(
        data_path, parse_dates=True
    )
    return df

def _clean_data(df:pd.DataFrame=None):
    init_row_count = len(df. index)
    # Drop any NaN values (if necessary)
    df.dropna()

    # Replace "normal" status with 0 and "anomaly" status with 1
    df.replace("normal", 0, inplace=True)
    df.replace("anomalous", 1, inplace=True)
        
    # Clean out any values in the top and bottom 1 percentiles
    for col in ["temperature", "pressure", "humidity"]:
        q_low = df[col].quantile(0.01)
        q_hi  = df[col].quantile(0.99)

        df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]
    removed_row_count = init_row_count - len(df_filtered.index)
    return df_filtered, removed_row_count

def _display(df):
    # Some path management
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'img/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    # Create plots to view tha data
    sns.boxplot(x = df["temperature"])
    plt.savefig(results_dir + "temp_boxplot.jpg")
    sns.displot(df["temperature"], bins=15, kde=False)
    plt.savefig(results_dir + "temp_displot.jpg")
        
    sns.boxplot(x = df["pressure"])
    plt.savefig(results_dir + "pressure_boxplot.jpg")
    sns.displot(df["pressure"], bins=15, kde=False)
    plt.savefig(results_dir + "pressure_displot.jpg")

    sns.boxplot(x = df["humidity"])
    plt.savefig(results_dir + "humidity_boxplot.jpg")
    sns.displot(df["humidity"], bins=15, kde=False)
    plt.savefig(results_dir + "humidity_displot.jpg")

"""
Function for retrieving a pandas data frame and clean it up

Input:
    data_path (str)     : path to csv data file
    plot_df   (bool)    : True: create plots of data; False: do not create plots of data

Output:
    df (pd.DataFrame)   : A clenaed pandas dataframe        
"""
def get_df_and_cleanup(data_path:str = "./ass2/ass2data.csv", plot_df:bool=False):
    if os.path.exists(data_path) and data_path.split(".")[-1] == "csv":
        print(f'Found data file at "{data_path}"!')
        df = _get_df(data_path=data_path)
        print(f'Shape of dataframe before cleaning: {df.shape}')
        print(f'Number of NaN values in data: {df.isna().sum().sum()}')
        if plot_df:
            _display(df)
        df, rm_rows = _clean_data(df)
        print(f'Shape of dataframe after cleaning: {df.shape}. Removed rows: {rm_rows}')
        return df
        

if __name__ == "__main__":
    df = get_df_and_cleanup("./ass2/ass2data.csv", True)
    print(df.head(2))