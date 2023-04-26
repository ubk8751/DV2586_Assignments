# Self-made modules
from AI import get_imgGPT
from data import create_data, remove
from eval import print_stat_row, get_confusion_matrix

# External libs
from keras.layers import *
from keras.callbacks import *
from keras.utils import to_categorical

# Global variables
tds_name  = "TrainingDataSet.tfds"
vds_name  = "ValidationDataSet.tfds"
data_path = "./ass1/MiniDIDAKeras.ds"
del_data  = True

# Run the program
if __name__ == "__main__":
    tds,vds = create_data(data_path)
    tds.save(tds_name)
    vds.save(vds_name)

    imggpt   = get_imgGPT()

    fitimggpt    = imggpt.fit(tds, valds=vds, epochs=10, batch_size=128)
    
    ImgGPT_stat = imggpt.evaluate(vds, 128)
    
    print("{: <15} {: <15} {: <15} {: <15}".format("Model", "Accuracy", "Loss",  "F1 score"))
    print_stat_row(ImgGPT_stat, "ImgGPT")
    if del_data:
        remove(tds_name)
        remove(vds_name)
