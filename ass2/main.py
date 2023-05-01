# Import packages
import os
import matplotlib.pyplot as plt

# Import modules
from data import get_df_and_cleanup

data_path = "./ass2/ass2data.csv"

def main(path:str = "./ass2/ass2data.csv", export_df:bool=False):
    df = get_df_and_cleanup(path, export_df)
    print(df.head(2))
    plot_column(df, "temperature")

def plot_column(df, col_name:str="temperature"):
    # Some path management
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'img/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    print(df.index)
    plt.plot(x = df.index, y=df[col_name])
    plt.legend()
    plt.savefig(results_dir + col_name + "_plot.png")


if __name__ == "__main__":
    main(data_path, False)