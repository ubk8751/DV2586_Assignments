# Import packages
import os
import matplotlib.pyplot as plt

# Import modules
from data import get_df_and_cleanup, get_train_test, plot_column, normalize_df, get_x_y
from model import AnomalyDetector

data_path = "./ass2/ass2data.csv"

def main(path:str = "./ass2/ass2data.csv", export_df:bool=False):
    df = get_df_and_cleanup(path, export_df, filter=False)
    train, test = get_train_test(df=df)
    train, test = normalize_df(train=train, test=test)
    LSTMAuto =  AnomalyDetector(train_data=train,opt="adam", loss="mae")
    xt, yt = get_x_y(train)
    xv, yv = get_x_y(test)
    LSTMHist = LSTMAuto.fit_model(xt=xt, yt=yt, xv=xv, yv=yv)


if __name__ == "__main__":
    main(data_path, False)