from keras.layers import Activation, BatchNormalization, Conv1D, Dense
from keras.layers import Flatten, MaxPooling1D
from keras.models import Sequential

from predictor import InceptionLayer
from .misc import add_dataset_info


def model_2(dataset):    
    model = Sequential(name='model_2')
    add_dataset_info(model, dataset)
    
    # inception layer 
    kernel_dict = {1:6, 3:6, 5:6, 7:6, 9:6}    
    model.add(InceptionLayer(kernel_dict, activation='relu', name='0_incept'))     
    
    # inception layer 
    layer_dict = {1:4, 3:4, 5:4, 7:4, 9:4}    
    model.add(InceptionLayer(layer_dict, activation='relu', name='1_incept'))     
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', 
                           name='1_max'))
    
    # network-in-network (1x1 convolution in 2D)
    model.add(Conv1D(filters=10, kernel_size=1, padding='valid',
                     activation=None, use_bias=False, name='2_net_in_net') )   
    model.add(BatchNormalization(name='2_norm'))    
    model.add(Activation('relu', name='2_relu'))    
    
    # dense
    model.add(Flatten(name='3_flat'))    
    model.add(Dense(20, use_bias=False, name='3_dense')) 
    model.add(BatchNormalization(name='3_norm'))    
    model.add(Activation('relu', name='3_relu'))
    
    # dense
    model.add(Dense(10, use_bias=False, name='4_dense'))
    model.add(BatchNormalization(name='4_norm'))    
    model.add(Activation('relu', name='4_relu'))
    
    # output softmax
    model.add(Dense(model.category_count, activation='softmax', 
                    name='5_softmax'))
    
    return model
    