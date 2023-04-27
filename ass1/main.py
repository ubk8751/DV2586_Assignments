# Self-made modules
from AI import get_imgGPT
from data import create_data, remove
from eval import print_stat_row, get_confusion_matrix
from PyAIGen import vgg19, ResNet50, DenseNet

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

    # Vgg_19, vgg_conf_matrix, vgg_TP, vgg_TN, vgg_FP, vgg_FN, vgg_F1        = vgg19(num_epochs=100, 
    #                                                                                 lr=0.001, 
    #                                                                                 momentum=0.9, 
    #                                                                                 step_size=7, 
    #                                                                                 gamma=0.1, 
    #                                                                                 v_model=False)
    # DN, DN_conf_matrix, DN_TP, DN_TN, DN_FP, DN_FN, DN_F1                  = DenseNet(num_epochs=100, 
    #                                                                                    lr=0.001, 
    #                                                                                    momentum=0.9, 
    #                                                                                    step_size=7, 
    #                                                                                    gamma=0.1, 
    #                                                                                    v_model=False)
    # RN50, RN50_conf_matrix, RN50_TP, RN50_TN, RN50_FP, RN50_FN, RN50_F1    = ResNet50(num_epochs=100, 
    #                                                                                    lr=0.001, 
    #                                                                                    momentum=0.9, 
    #                                                                                    step_size=7, 
    #                                                                                    gamma=0.1, 
    #                                                                                    v_model=False)

    imggpt      = get_imgGPT()
    fitimggpt   = imggpt.fit(tds, valds=vds, epochs=5, batch_size=128)
    ImgGPT_stat = imggpt.evaluate(valds=vds,hist=fitimggpt,model=imggpt,batch_size=128)
    print(ImgGPT_stat)
    # print("{: <15} {: <15} {: <15} {: <15}".format("Model", "Accuracy", "Loss",  "F1 score"))
    # print_stat_row(ImgGPT_stat, "ImgGPT")
    if del_data:
        remove(tds_name)
        remove(vds_name)