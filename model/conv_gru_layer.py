from keras.layers import Activation, BatchNormalization, Conv1D, Concatenate
from keras.layers import GRU, Layer

from model.add_dataset_info import add_dataset_info


class ConvGRULayer(Layer):
    
    
    def __init__(self, kernel_dict, gru_size, activation, name):        
        super().__init__()         
        self.layers = []        
        
        for kernel_size, filters in kernel_dict.items():
            layer = Conv1D(filters=filters, 
                           kernel_size=kernel_size,
                           padding='same',
                           activation=None,
                           use_bias=False,
                           name=name+'_conv_'+str(kernel_size))            
            self.layers.append(layer)   
            
        layer = GRU(units=gru_size, 
                    activation=None,
                    use_bias=False,
                    return_sequences=True,
                    name=name+'_gru_'+str(gru_size))
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

