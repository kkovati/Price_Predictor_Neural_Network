from keras.layers import Activation, BatchNormalization, Conv1D, Dense
from keras.layers import Flatten, GRU, MaxPooling1D
from keras.models import Sequential




def model_3(dataset):    
    model = Sequential(name='model_3')
    
    # add dataset info fields into model
    model.input_interval = dataset.input_interval
    model.prediction_interval = dataset.prediction_interval
    model.categories = dataset.categories
    
    
    model.add(GRU(10, return_sequences=True))
    
    model.add(Flatten())
    
    # output softmax
    category_count = len(dataset.categories) + 1
    model.add(Dense(category_count, activation='softmax', name='4_softmax'))
    
    return model
    

# =============================================================================
# import keras
# x_in = keras.layers.Input((30,4))
# 
# print(x_in.shape)
# 
# x = keras.layers.GRU(10, return_sequences=True)(x_in)
# 
# print(x.shape)
# 
# x = Dense(4, activation='softmax', name='4_softmax')(x)
# 
# print(x.shape)
# 
# m = keras.models.Model(inputs=x_in, outputs=x)
# =============================================================================

