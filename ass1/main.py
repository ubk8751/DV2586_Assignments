# Self-made modules
from AI import get_imgGPT, fit_model, get_models, evaluate
from data import create_data_set, create_tds, create_vds, remove

# External libs
from keras.layers import *
from keras.callbacks import *

# Global variables
tds_name  = "TrainingDataSet.tfds"
vds_name  = "ValidationDataSet.tfds"
data_path = "./ass1/MiniDIDA"
del_data  = True

# Run the program
if __name__ == "__main__":
    # Create the data sets
    X_train, X_test, y_train, y_test = create_data_set(data_path, rs=42, ts=0.2, num_classes=10)
    
    # Turn them into tensorflow datasets
    tds = create_tds(X_train=X_train, y_train=y_train, tds_name="TrainingDataSet.tfds", buffer_size=10, batches=2)
    vds = create_vds(X_test=X_test, y_test=y_test, vds_name="ValidationDataSet.tfds", buffer_size=10, batches=2)
    
    #Train pre-trained models
    vgg, densenet, resnet = get_models()

    # Fit pre-trained models
    fit_models = {}
    for mod in [vgg, densenet, resnet]:
        fit_models[mod] = fit_model(mod, X_train, y_train, X_test, y_test)

    # Create the ultimate image recognition AI architecture
    imggpt = get_imgGPT()

    # Fit the ultimate image recognition AI architecture
    fitimggpt = imggpt.fit(trainds=tds, valds=vds, epochs=20, batch_size=128)

    # Evaluate pre-trained models
    vgg_stat        = evaluate(vgg, fit_models[vgg], X_test, y_test)
    densenet_stat   = evaluate(densenet, fit_models[densenet], X_test, y_test)
    resnet_stat     = evaluate(resnet, fit_models[resnet], X_test, y_test)
    
    if del_data:
        remove(tds_name)
        remove(vds_name)
