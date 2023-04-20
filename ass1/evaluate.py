import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# Evaluate model
def evaluate(model, hist, xv, yv):
    ret = {
        "score":    model.evaluate(xv, yv),
        "accuracy": hist.history["accuracy"], 
        "val_acc":  hist.history["val_accuracy"],
        "loss":     hist.history["loss"],
        "val_loss": hist.history["val_loss"]
    }
    return ret

def get_confusion_matrix(y_train, y_pred, normalize:bool=True):
    pass

# def _f1_score(model):
#     precision_score
#     return 2*((precision*recall)/(precision+recall))
    