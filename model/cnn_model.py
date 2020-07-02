from keras.layers import Input 
from keras.layers import BatchNormalization, LayerNormalization 
from keras.layers import Conv1D
from keras.models import Model


def cnn_model(input_shape):    
   
    X_input = Input(input_shape)     
    
    print(X_input.shape)
    
    X = LayerNormalization(axis=1)
    
    X = Conv1D(filters=32, 
               kernel_size=2,
               strides=1,
               #padding=,
               activation='relu',
               use_bias=True,
               name = 'conv0')(X_input)
    
    print(X.shape)
    
    
    
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
# =============================================================================
#     X = Activation('relu')(X)
# 
#     # MAXPOOL
#     X = MaxPooling2D((2, 2), name='max_pool')(X)
# 
#     # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
#     X = Flatten()(X)
#     X = Dense(1, activation='sigmoid', name='fc')(X)
# =============================================================================
    
    return Model(inputs = X_input, outputs = X, name='CNNModel')

def norm_model(input_shape):
    
    X_input = Input(input_shape)     
    
    print(X_input.shape)
        
    X = LayerNormalization(axis=1)
    
model = norm_model((6,4))

model.summary()