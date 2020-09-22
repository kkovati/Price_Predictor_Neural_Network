from keras.layers import Activation, BatchNormalization, Conv1D, Concatenate
from keras.layers import Layer


class InceptionLayer(Layer):
    """
    Combination of multiple 1D Convolutional layers.
    The Convolutional layers can have different kernel sizes, because all are 
    using 'same' padding, thus all of them can be concatecated.
    """    
    def __init__(self, kernel_dict, activation, name):  
        """
        Parameters:
        kernel_dict : dict : {key : int, value : int}
            Dictionary which contains kernel sizes and filter numbers. Key is 
            the kernel_size, value is the number of the filters.
        activation : str
            Activation function of the whole layers
        name : str
            Name of layer
        """
        super().__init__()         
        self.layers = []        
        
        # Conv1D layers
        for kernel_size, filters in kernel_dict.items():
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


