# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

# Import stuff fro mmy own modules
from data import get_data, _sequencify_df
from model import AnomalyDetector

# Probably can go without since I only use it in one place, but you never know...
data_path = "./ass2/dataset2.csv"

def main(path:str = "./ass2/dataset2.csv", export_df:bool=False):
    # Get data, plit into datasets
    (train, y_train), (val, y_val), scalers, df = get_data(path=path, export_df=export_df)
    
    # Create model
    LSTMAO =  AnomalyDetector(train_data=train,opt="adam", loss="mae")
    
    # Fit model
    LSTMHist = LSTMAO.fit_model(train=train,validation=val, epochs=5)
    plot_loss(hist=LSTMHist.history, labels=['loss', 'val_loss'], title='Model Loss', y_label='loss (MAE)', x_label='epoch')

    # Evaluate model
    e_reconstruct = LSTMAO.create_thresholds(val=val, data_actual=val, batch_size=4)
    print(f'Reconstruction errors:')
    for i, item in enumerate(e_reconstruct):
        print(f'Bearing {i+1} Reconstruction error: {e_reconstruct[i]}')

    # Setup to plot the reconstruction data and anomalies
    sequence_list = _sequencify_df(df=df, sl=5)
    date_list = mdates.datestr2num(df['timestamp'])
    
    # Plot the results. Hardcoding column names because the amount of retunr values from  get_data is getting ridiculous...
    plot_results(sequence_list=sequence_list, date_list=date_list, data_cols=['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'], scalers=scalers, model=LSTMAO, reconstruction_errors=e_reconstruct)

# Since loss is the only "metric" i get from the model, it can be interesting to plot out
def plot_loss(hist, labels:list=['loss', 'val_loss'], title:str='Temp', x_label:str='x_temp', y_label:str='y_temp', save_name:str='lossplot.jpg'):
    fig, ax = plt.subplots(figsize=(14,6),dpi=80)
    for label in labels:
        ax.plot(hist[label], label=label, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.legend(loc="upper right")
    ax.set_title("Sensor data")
    plt.savefig("./ass2/img/" + save_name)

def plot_results(sequence_list, date_list:list, data_cols:list, scalers:list, model:AnomalyDetector, reconstruction_errors:list):
    # Recreate data to plot
    data = sequence_list[:,0,:]
    data = np.concatenate([data, sequence_list[-1,1:,:]])
    
    # Create recreation (predict on data)
    preds = model._predict_model(sequence_list, batch_size=4)
    e = np.abs(preds - sequence_list)
    error = e[0,:,:]
    error = np.concatenate([error,e[1:,-1,:]])

    # Create reconstructed data to plot
    init = preds[0,:,:]
    r_data = preds[1:,-1,:]
    r_data = np.concatenate([init, r_data])

    # Create dataframe
    df = pd.DataFrame(error, columns = [col + "_mae" for col in data_cols])
    for i, col in enumerate(data_cols):
        df[col] = data.T[i]
        df['r_' + col] = r_data.T[i] = r_data.T[i]
        df[col + ' Threshold'] = reconstruction_errors[i]
        df[col + ' Anomaly'] = df[col + '_mae'] > df[col + ' Threshold']
    df['timestamp'] = mdates.num2date(date_list)

    # If we have scalers, use them to normalize our dataframe (pretty much repeat of _normalize_data from data.)
    if scalers:
        for i, col in enumerate(data_cols):
            df[[col]] = scalers[0].inverse_transform(df[[col]])
            df[['r_' + col]] = scalers[0].inverse_transform(df[['r_' + col]])
            df[[col + ' Threshold']] = scalers[0].inverse_transform(df[[col + ' Threshold']])

    # PLot time
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Reconstruction with plotted anomalies')
    for i, col in enumerate(data_cols):
        ax = plt.subplot(2, 2, i+1)
        plt.xticks(rotation=30)
        plt.title(col)

        # Plot the real and predicted data/reconstruction
        plt.plot(date_list, df[col], label='Real data', linewidth=1)
        plt.plot(date_list, df['r_' + col], linewidth=1, linestyle='dashed', label='Reconstructed')
        
        # Plot hte anomlaies
        anomalies = df[df[col + ' Anomaly'] == True]
        plt.scatter(
            anomalies['timestamp'], 
            anomalies[col],
            marker='.',
            c='red',
            s=4,
            label='Anomaly',
            zorder=0)
        plt.legend()

        ax.xaxis_date()
        ax.set_xticks(date_list[::100],)
        ax.set_xticklabels(date_list[::100])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.savefig("./ass2/img/anomalies.jpg")
    print("Done!")

if __name__ == "__main__":
    main(data_path, False)