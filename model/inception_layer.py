from keras.layers import Conv1D

class InceptionLayer:
    
    def __init__(self, 1_filter):
        conv_1 = Conv1D(filters=32, 
                        kernel_size=1,
                        padding='same',
                        activation='relu',
                        use_bias=True,
                        name = 'conv0??')
        
        conv_3 = Conv1D(filters=32, 
                        kernel_size=3,
                        padding='same',
                        activation='relu',
                        use_bias=True,
                        name = 'conv0??')

    def __call__(self, X_input):
        pass