import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# Evaluate model
def evaluate(model, hist, xv, yv):
    ret = {
        "score":    model.evaluate(xv, yv),
        "accuracy": hist.history["accuracy"][-1], 
        "val_acc":  hist.history["val_accuracy"][-1],
        "loss":     hist.history["loss"][-1],
        "val_loss": hist.history["val_loss"][-1],
        "f1_score": _f1_score(hist=hist)
    }
    return ret

def get_confusion_matrix(hist, y_train, y_pred, normalize:bool=True):
    pass

def _f1_score(hist):
    tp = hist.history["true_positives"][-1]
    fp = hist.history["false_positives"][-1]
    tn = hist.history["true_negatives"][-1]
    fn = hist.history["false_negatives"][-1]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return 2*((precision*recall)/(precision+recall))
    