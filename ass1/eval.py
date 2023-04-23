import numpy as np
from sklearn.metrics import confusion_matrix

# Evaluate model
def evaluate(hist, model, xv, yv):
    ret = {
        "eval":     model.evaluate(xv , yv, batch_size=128),
        "accuracy": hist.history["accuracy"][-1], 
        "val_acc":  hist.history["val_accuracy"][-1],
        "loss":     hist.history["loss"][-1],
        "val_loss": hist.history["val_loss"][-1],
        "f1_score": _f1_score(hist=hist)
    }
    return ret

def get_confusion_matrix(model, x_test, y_test):
    #Predict
    y_prediction = model.predict(x_test)
    y_prediction = np.argmax (y_prediction, axis = 1)
    y_test=np.argmax(y_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    result = confusion_matrix(y_test, y_prediction , normalize='pred')
    return result

def _f1_score(hist):
    tp = hist.history["true_positives"][-1]
    fp = hist.history["false_positives"][-1]
    tn = hist.history["true_negatives"][-1]
    fn = hist.history["false_negatives"][-1]
    if tp == 0.0 or fp == 0.0:    
        if tp == 0.0:
            print("Model has 0 true positives")
        if fp == 0.0:
            print("Model has 0 false positives")
        if tp == 0.0 and fp == 0.0:
            return 0
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return 2*((precision*recall)/(precision+recall))

def print_stat_row(stat, name):
    print(f'{name: <15} {stat["accuracy"]:<15.5f} {stat["loss"]:<15.5f} {stat["f1_score"]:<15.5f}'.format(name,stat["accuracy"],stat["loss"],stat["f1_score"]))
    