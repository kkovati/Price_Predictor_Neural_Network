import unittest
from keras.layers import Input
from keras.models import Model

from predictor import InceptionLayer


class TestInceptionLayer(unittest.TestCase):
    
    def test_inception_layer(self):
        print('\n---Run test_inception_layer---')
        layer_dict = {1: 12, 3: 14, 5: 16}
        
        incept_layer = InceptionLayer(layer_dict, activation='relu', 
                                      name='test_layer')
        
        X_input = Input((30, 4))    
        print('Input shape:', X_input.shape)
        
        X = incept_layer(X_input)
        print('Output shape:', X.shape)    
        
        # input interval must remain the same
        assert X.shape[1] == 30
        # no. of channels add up: 42 == 12 + 14 + 16
        assert X.shape[2] == 42
        
        model = Model(inputs=X_input, outputs=X, name='TestModel')    
        model.summary()

if __name__ == '__main__':
    unittest.main()   