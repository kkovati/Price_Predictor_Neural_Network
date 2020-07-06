from keras.layers import Input
from keras.models import Model

from model import InceptionLayer


if __name__ == '__main__':
    
    layer_dict = {1: 12, 3: 14, 5: 16}
    
    incept_layer = InceptionLayer(layer_dict, name='test_layer')
    
    X_input = Input((30, 4))    
    print('Input shape:', X_input.shape)
    
    X = incept_layer(X_input)
    print('Output shape:', X.shape)    
    
    model = Model(inputs=X_input, outputs=X, name='TestModel')    
    model.summary()
