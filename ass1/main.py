# Self-made modules
from AI import get_imgGPT, fit_model, get_models, evaluate, get_vgg, get_resnet, get_densenet
from data import create_data_set, create_tds, create_vds, remove
from eval import print_stat_row, get_confusion_matrix

# External libs
from keras.layers import *
from keras.callbacks import *

# Global variables
tds_name  = "TrainingDataSet.tfds"
vds_name  = "ValidationDataSet.tfds"
data_path = "./ass1/MiniDIDA.ds"
del_data  = True

# Run the program
if __name__ == "__main__":
    # Create the data sets
    X_train, X_test, y_train, y_test = create_data_set(data_path, rs=42, ts=0.2, num_classes=10)
    
    # Turn them into tensorflow datasets
    tds = create_tds(X_train=X_train, y_train=y_train, tds_name="TrainingDataSet.tfds", buffer_size=10, batches=2)
    vds = create_vds(X_test=X_test, y_test=y_test, vds_name="ValidationDataSet.tfds", buffer_size=10, batches=2)
    
    #Train pre-trained models
    vgg      = get_vgg()
    densenet = get_densenet() 
    resnet   = get_resnet()

    # Fit pre-trained models
    fit_vgg      = fit_model(vgg, X_train, y_train, X_test, y_test)
    fit_densenet = fit_model(densenet, X_train, y_train, X_test, y_test)
    fit_resnet   = fit_model(resnet, X_train, y_train, X_test, y_test)

    # VGG Confusion matrix
    vgg_cm = get_confusion_matrix(vgg, x_test=X_test, y_test=y_test)
    print(vgg_cm)
    exit()
    # Create the ultimate image recognition AI architecture
    imggpt = get_imgGPT()

    # Fit the ultimate image recognition AI architecture
    fitimggpt = imggpt.fit(trainds=tds, valds=vds, epochs=20, batch_size=128)

    # Evaluate pre-trained models
    vgg_stat        = evaluate(fit_vgg, vgg, X_train, y_train)
    densenet_stat   = evaluate(fit_densenet, densenet, X_train, y_train)
    resnet_stat     = evaluate(fit_resnet, resnet, X_train, y_train)
    ImgGPT_stat     = evaluate(fitimggpt, imggpt, X_train, y_train)
    
    print("{: <15} {: <15} {: <15} {: <15}".format("Model", "Accuracy", "Loss",  "F1 score"))
    print_stat_row(vgg_stat, "VGG-19")
    print_stat_row(densenet_stat, "DenseNet")
    print_stat_row(resnet_stat, "ResNet50")
    print_stat_row(ImgGPT_stat, "ImgGPT")
    if del_data:
        remove(tds_name)
        remove(vds_name)
