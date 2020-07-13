from keras.layers import Activation, BatchNormalization, Conv1D, Dense
from keras.layers import Flatten, Input, MaxPooling1D
from keras.models import Sequential

from model.inception_layer import InceptionLayer


def model_1(dataset):    
    model = Sequential(name='model_1')   
    
    # add dataset info fields into model
    model.input_interval = dataset.input_interval
    model.prediction_interval = dataset.prediction_interval
    model.categories = dataset.categories 
       
    # inception layer    
    layer_dict = {1:10, 3:10, 5:10, 7:10}    
    model.add(InceptionLayer(layer_dict, activation='relu', name='0_incept'))     
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', 
                           name='0_pool'))
    
    # network-in-network (1x1 convolution in 2D)
    model.add(Conv1D(filters=10, kernel_size=1, padding='valid',
                     activation=None, use_bias=False, name='1_net_in_net') )   
    model.add(BatchNormalization(name='1_norm'))    
    model.add(Activation('relu', name='1_relu'))    
    
    # conversion to fully connected
    model.add(Flatten(name='2_flat'))    
    model.add(Dense(20, use_bias=False, name='2_dense')) 
    model.add(BatchNormalization(name='2_norm'))    
    model.add(Activation('relu', name='2_relu'))
    
    # normal fully connected
    model.add(Dense(10, use_bias=False, name='3_dense'))
    model.add(BatchNormalization(name='3_norm'))    
    model.add(Activation('relu', name='3_relu'))
    
    # output softmax
    category_count = len(dataset.categories) + 1
    model.add(Dense(category_count, activation='softmax', name='4_softmax'))
    
    return model

