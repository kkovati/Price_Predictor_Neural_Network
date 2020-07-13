from keras.layers import Activation, BatchNormalization, Conv1D, Dense
from keras.layers import Flatten, MaxPooling1D
from keras.models import Sequential

from model.inception_layer import InceptionLayer


def model_3(dataset):    
    model = SequentialPredictor(dataset=dataset, name='model_3')
    
    # output softmax
    model.add(Dense(4, activation='softmax', name='5_softmax', input_shape=(30,4)))
    
    return model
    