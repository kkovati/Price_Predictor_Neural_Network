from keras.layers import Activation, BatchNormalization, Conv1D, Dense
from keras.layers import Flatten, MaxPooling1D
from keras.models import Sequential

from predictor import ConvGRULayer
from predictor import add_dataset_info


def model_3(dataset):    
    model = Sequential(name='model_3')
    add_dataset_info(model, dataset)
    
    # conv_gru layer 
    kernel_dict = {1:8, 3:8, 5:8, 7:8, 9:8}    
    gru_size = 10
    model.add(ConvGRULayer(kernel_dict, gru_size, activation='relu', 
                           name='0_conv_gru'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', 
                           name='0_max'))
    
    # network-in-network (1x1 convolution in 2D)
    model.add(Conv1D(filters=10, kernel_size=1, padding='valid',
                     activation=None, use_bias=False, name='1_net_in_net') )   
    model.add(BatchNormalization(name='1_norm'))    
    model.add(Activation('relu', name='1_relu'))    
    
    # dense
    model.add(Flatten(name='2_flat'))    
    model.add(Dense(20, use_bias=False, name='2_dense')) 
    model.add(BatchNormalization(name='2_norm'))    
    model.add(Activation('relu', name='2_relu'))
    
    # dense
    model.add(Dense(10, use_bias=False, name='3_dense'))
    model.add(BatchNormalization(name='3_norm'))    
    model.add(Activation('relu', name='3_relu'))
    
    # output softmax
    model.add(Dense(model.category_count, activation='softmax', 
                    name='4_softmax'))
    
    return model
    



