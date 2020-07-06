from keras.layers import Conv1D, Concatenate


class InceptionLayer:
    
    def __init__(self, layer_dict, name):
        
        self.layers = []        
        
        for kernel_size, filters in layer_dict.items():
            layer = Conv1D(filters=filters, 
                           kernel_size=kernel_size,
                           padding='same',
                           activation='relu',
                           use_bias=True,
                           name=name+'_conv_'+str(kernel_size))            
            self.layers.append(layer)        
        
        self.concat = Concatenate(axis=2, name=name+'_concat')

    def __call__(self, X_input):
        
        X_outputs = [layer(X_input) for layer in self.layers]        
            
        return self.concat(X_outputs)


