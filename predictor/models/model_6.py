from keras.layers import Activation, BatchNormalization, Conv1D, Dense
from keras.layers import Flatten, GRU, MaxPooling1D
from keras.models import Sequential

from predictor import ConvGRULayer
from .misc import add_dataset_info


def model_6(dataset):    
    model = Sequential(name='model_6')
    add_dataset_info(model, dataset)
    
    # conv_gru layer 
    kernel_dict = {1:8, 3:8, 5:8, 7:8, 9:8}    
    gru_size = 10
    model.add(ConvGRULayer(kernel_dict, gru_size, activation='relu', 
                           name='0_conv_gru'))
    
    # conv_gru layer 
    kernel_dict = {1:6, 3:6, 5:6, 7:6, 9:6}    
    gru_size = 8
    model.add(ConvGRULayer(kernel_dict, gru_size, activation='relu', 
                           name='1_conv_gru'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', 
                           name='1_max'))
    
    # network-in-network (1x1 convolution in 2D)
    model.add(Conv1D(filters=10, kernel_size=1, padding='valid',
                     activation=None, use_bias=False, name='2_net_in_net') )   
    model.add(BatchNormalization(name='2_norm'))    
    model.add(Activation('relu', name='2_relu'))    
    
    # GRU (returns only last last output)
    model.add(GRU(units=10, activation=None, use_bias=False, 
                  return_sequences=False, name='3_gru'))
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
    



