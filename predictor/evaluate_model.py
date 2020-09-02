import keras
from keras import backend as K
import numpy as np

from dataset_generation import Dataset


"""
This script is used to test trained models saved into .h5 files
It loads a test dataset, loads a model and evaluates
"""
if __name__ == '__main__':
    
    #dataset_name = '../dataset_generation/datasets_npz/IN60PRED1CAT[1,5].npz'
    dataset_name = '../dataset_generation/datasets_npz/test_dataset_IN3PRED2CAT[1,2,3].npz'
    dataset = Dataset.load(dataset_name)
    
    test_input_set, test_label_set =  dataset.get_test_set()
    
    path = 'trained_models_h5/'
# =============================================================================
#     modelfiles = ['model_1_IN60PRED1CAT[1,5]_5epo.h5',
#                   'model_2_IN60PRED1CAT[1,5]_5epo.h5',
#                   'model_3_IN60PRED1CAT[1,5]_1epo.h5',
#                   'model_4_IN60PRED1CAT[1,5]_1epo.h5']
# =============================================================================
    modelfiles = ['model_4_IN3PRED2CAT[1,2,3].h5']
    
    for mf in modelfiles:    
        print('asd')
        model = keras.models.load_model(path + mf)
        
        print('---------------------------------------------')
        print('Testing of', model.name)
        
        loss = model.evaluate(test_input_set, test_label_set)
        
        print('Model metrics and losses!!!:')
        print(model.metrics_names)
        print(loss) 
        
        trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    
        print('Total params: {:,}'.format(trainable_count + 
                                          non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    
