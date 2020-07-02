import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler

from dataset_generation import generate


class TestStandardScaler(unittest.TestCase):
    
    def test_standard_scaler(self):
        print('\nRun test_standard_scaler')

        arr = [[100, 110], [200, 220], [300, 330], [400, 440], [500, 550]]
        print(arr)
        arr = np.array(arr)
        

# =============================================================================
#         train_set = generate(input_interval=6, 
#                              prediction_interval=2, 
#                              categories=[1,2.5,5],
#                              training_set_size=1, 
#                              test_set_size=1,
#                              train_data='test_data/test_crypto-markets.csv', 
#                              test_data='test_data/test_crypto-markets.csv')
#         
#         input_set, label_set = train_set
# =============================================================================
        
        scaler = StandardScaler()
        
        standardized_arr = scaler.fit_transform(arr.flatten().reshape(1, -1).transpose())
        
        print(standardized_arr)
        
        
        sta_arr = (arr - np.mean(arr)) / np.std(arr)
        
        print(sta_arr)
        
        

        
        
if __name__ == '__main__':
    unittest.main()        