# Self-made modules
from imgGPT import get_imgGPT
from data import create_data, remove, print_stat_row
from PyAIGen import vgg19, ResNet50, DenseNet
import matplotlib.pyplot as plt

# External libs
from keras.layers import *
from keras.callbacks import *

# Global variables
tds_name  = "TrainingDataSet.tfds"
vds_name  = "ValidationDataSet.tfds"
#data_path = "./ass1/MiniDIDA.ds"
data_path = "./ass1/DIDA.ds"
del_data  = True
epochs = 10

# Run the program
if __name__ == "__main__":
    tds,vds = create_data(data_path)
    tds.save(tds_name)
    vds.save(vds_name)

    imggpt      = get_imgGPT()
    fitimggpt   = imggpt.fit(tds, valds=vds, epochs=epochs, batch_size=1024)
    ImgGPT_stat = imggpt.vgg_evaluate(valds=vds,hist=fitimggpt,model=imggpt)

    Vgg_19, vgg_stats = vgg19(num_epochs=epochs, 
                              lr=0.001, 
                              momentum=0.9, 
                              step_size=7, 
                              gamma=0.1, 
                              v_model=False)
    DN, DN_stats      = DenseNet(num_epochs=epochs, 
                                 lr=0.001, 
                                 momentum=0.9, 
                                 step_size=7, 
                                 gamma=0.1, 
                                 v_model=False)
    RN50, RN50_stats  = ResNet50(num_epochs=epochs, 
                                 lr=0.001, 
                                 momentum=0.9, 
                                 step_size=7, 
                                 gamma=0.1, 
                                 v_model=False)

    
    print("{: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}".format("Model", "Accuracy", "Val_Acc", "Loss", "Val_Loss", "TP", "FP", "TN", "FN", "F1 score"))
    print_stat_row(ImgGPT_stat, "ImgGPT")
    print_stat_row(vgg_stats, "VGG-19")
    print_stat_row(RN50_stats, "ResNet50")
    print_stat_row(DN_stats, "DenseNet")

    acc = fitimggpt.history["accuracy"]
    val_acc = fitimggpt.history["val_accuracy"]
    loss = fitimggpt.history["loss"]
    val_loss = fitimggpt.history["val_loss"]

    x_axis = list(range(1, epochs+1))
    fig, ax = plt.subplots()
    ax.plot(x_axis, acc, color="blue", label="Train")
    ax.plot(x_axis, val_acc, color="red", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_yticks(np.arange(0, 1,step=0.1))
    ax.grid(True, which="both")
    plt.savefig("Accuracy.jpg")

    x_axis = list(range(1, 5+1))
    fig, ax = plt.subplots()
    ax.plot(x_axis, loss, color="blue", label="Train")
    ax.plot(x_axis, val_loss, color="red", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_yticks(np.arange(0, max(val_loss)))
    ax.grid(True, which="both")
    plt.savefig("Loss.jpg")

    if del_data:
        remove(tds_name)
        remove(vds_name)