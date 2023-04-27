# Self-made modules
from imgGPT import get_imgGPT
from data import create_data, remove, print_stat_row
from PyAIGen import vgg19, ResNet50, DenseNet

# External libs
from keras.layers import *
from keras.callbacks import *

# Global variables
tds_name  = "TrainingDataSet.tfds"
vds_name  = "ValidationDataSet.tfds"
data_path = "./ass1/DIDA.ds"
del_data  = True

# Run the program
if __name__ == "__main__":
    tds,vds = create_data(data_path)
    tds.save(tds_name)
    vds.save(vds_name)

    Vgg_19, vgg_stats = vgg19(num_epochs=50, 
                              lr=0.001, 
                              momentum=0.9, 
                              step_size=7, 
                              gamma=0.1, 
                              v_model=False)
    DN, DN_stats      = DenseNet(num_epochs=50, 
                                 lr=0.001, 
                                 momentum=0.9, 
                                 step_size=7, 
                                 gamma=0.1, 
                                 v_model=False)
    RN50, RN50_stats  = ResNet50(num_epochs=50, 
                                 lr=0.001, 
                                 momentum=0.9, 
                                 step_size=7, 
                                 gamma=0.1, 
                                 v_model=False)

    imggpt      = get_imgGPT()
    fitimggpt   = imggpt.fit(tds, valds=vds, epochs=50, batch_size=128)
    ImgGPT_stat = imggpt.vgg_evaluate(valds=vds,hist=fitimggpt,model=imggpt,batch_size=128)
    print("{: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}".format("Model", "Accuracy", "Val_Acc", "Loss", "Val_Loss", "TP", "FP", "TN", "FN", "F1 score"))
    print_stat_row(ImgGPT_stat, "ImgGPT")
    print_stat_row(vgg_stats, "VGG-19")
    print_stat_row(RN50_stats, "ResNet50")
    print_stat_row(DN_stats, "DenseNet")
    if del_data:
        remove(tds_name)
        remove(vds_name)