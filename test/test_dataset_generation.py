import unittest
import pytest
import numpy as np

from dataset_generation import generate, standardize


class TestDatasetGeneration(unittest.TestCase):
    
    def generate_train_set(self):
        return generate(input_interval=6, 
                        prediction_interval=2, 
                        categories=[1,2.5,5],
                        training_set_size=10, 
                        test_set_size=1,
                        train_data='test_data/test_crypto-markets.csv', 
                        test_data='test_data/test_crypto-markets.csv')
    
    
    def test_generate(self):
        print('\nRun test_generate')
        
        input_set, label_set = self.generate_train_set()
        
        assert input_set[0][4][0] == 116.38
        assert label_set[0][2] == 0
        assert label_set[0][3] == 1
        
        print(input_set[0])
        print(label_set[0])
        
            
    def test_standardize(self):
        print('\nRun test_standardize')
        
        input_set, _ = self.generate_train_set()
                
        standarized_set = standardize(input_set)
        
        assert np.mean(standarized_set[0]) == pytest.approx(0)
        assert np.std(standarized_set[0]) == pytest.approx(1)
        
        print(input_set[0])
        print(standarized_set[0])        
        
        
if __name__ == '__main__':
    unittest.main()        
