# Import packages
import os
import matplotlib.pyplot as plt

# Import modules
from data import get_data
from model import AnomalyDetector

data_path = "./ass2/dataset2.csv"

def main(path:str = "./ass2/dataset2.csv", export_df:bool=False):
    train, val = get_data(path=path, export_df=export_df)
    LSTMAuto =  AnomalyDetector(train_data=train,opt="adam", loss="mae")
    
    LSTMHist = LSTMAuto.fit_model(train=train,validation=val, epochs=20)

    preds = LSTMAuto.predict_model(data=val, batch_size=4)


if __name__ == "__main__":
    main(data_path, False)