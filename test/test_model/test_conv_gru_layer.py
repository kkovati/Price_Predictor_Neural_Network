import unittest
from keras.layers import Input
from keras.models import Model

from model import ConvGRULayer


class TestConvGRULayer(unittest.TestCase):
    
    def test_convgru_layer(self):
        print('\n---Run test_convgru_layer---')
        kernel_dict = {1: 12, 3: 14, 5: 16}
        gru_size = 18
        
        conv_gru_layer = ConvGRULayer(kernel_dict, gru_size, activation='relu', 
                                    name='test_layer')
        
        X_input = Input((30, 4))    
        print('Input shape:', X_input.shape)
        
        X = conv_gru_layer(X_input)
        print('Output shape:', X.shape)    
        
        # input interval must remain the same
        assert X.shape[1] == 30
        # no. of channels add up: 42 == 12 + 14 + 16 + 18
        assert X.shape[2] == 60
        
        model = Model(inputs=X_input, outputs=X, name='TestModel')    
        model.summary()

if __name__ == '__main__':
    unittest.main()   