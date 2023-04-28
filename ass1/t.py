from imgGPT import get_imgGPT
from keras.utils.vis_utils import plot_model, model_to_dot
model = get_imgGPT()
model.build()
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)