from keras.layers import *
from keras.callbacks import *
from keras.models import Sequential
import eval
from imgGPT import ImgGPT

# Evaluate model
def evaluate(hist, model, valds):
    ret = eval.evaluate(hist, model, valds)
    return ret

# Fit model
def fit_model(mod, xt, xv, yt, yv, epochs:int=20, batch_size:int=128):
    fitmod = mod.fit(xt,yt, epochs=epochs, batch_size=batch_size, validation_data=(xv,yv))
    return fitmod

def get_imgGPT():
    print("\nCreating imgGPT")
    mod = Sequential()
    model = ImgGPT(mod, input_shape=(64,64,1))
    model.model.add(Flatten())
    model.model.add(Dense(10, activation='softmax'))
    model.compile(opt='sgd', loss="categorical_crossentropy")
    return model