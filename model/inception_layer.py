from keras.layers import Activation, BatchNormalization, Conv1D, Concatenate
from keras.layers import Layer


class InceptionLayer(Layer):
    
    def __init__(self, layer_dict, activation, name):        
        super().__init__()         
        self.layers = []        
        
        for kernel_size, filters in layer_dict.items():
            layer = Conv1D(filters=filters, 
                           kernel_size=kernel_size,
                           padding='same',
                           activation=None,
                           use_bias=False,
                           name=name+'_conv_'+str(kernel_size))            
            self.layers.append(layer)        
        
        self.concat = Concatenate(axis=2, name=name+'_concat')        
        self.batch_norm = BatchNormalization(axis=2, name=name+'_norm')        
        self.activate = Activation(activation, name=name+'_'+activation)

    def __call__(self, X_input):        
        X_list = [layer(X_input) for layer in self.layers]             
        
        X = self.concat(X_list)        
        X = self.batch_norm(X)        
        X = self.activate(X)        
        return X


