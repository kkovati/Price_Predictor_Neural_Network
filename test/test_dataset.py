import unittest
import pytest
import numpy as np

from dataset_generation import Dataset


class TestDataset(unittest.TestCase):
    
    def test_parse_csv(self):        
        print('\n---Run test_parse_csv---')
        dataset = Dataset()
    
        csv_list = dataset.parse_csv('test_data/test_crypto-markets_8_lines.csv')
        
        print(csv_list)
    
    def test_generate_training_set(self):
        print('\n---Run test_generate_training_set---')
        
        dataset = Dataset()      
        
        training_set = dataset.generate_training_set(
            input_interval=6, 
            prediction_interval=2, 
            categories=[1,2.5,5],
            set_size=1, 
            train_data='test_data/test_crypto-markets_8_lines.csv')

        input_set, label_set = training_set
        
        assert input_set[0][4][0] == 116.38
        assert label_set[0][2] == 0
        assert label_set[0][3] == 1
        
        print(input_set[0])
        print(label_set[0])
        
    def test_generate_test_set(self):
        print('\n---Run test_generate_test_set---')
        
        dataset = Dataset()      
        
        training_set = dataset.generate_training_set(
            input_interval=6, 
            prediction_interval=2, 
            categories=[1,2.5,5],
            set_size=1, 
            train_data='test_data/test_crypto-markets_8_lines.csv')        
        
        test_set = dataset.generate_test_set(
            set_size=2,
            test_data='test_data/test_crypto-markets_9_lines.csv')
        
        input_set, label_set = test_set
        
        print(input_set)
        print(label_set)
        
            
# =============================================================================
#     def test_standardize(self):
#         print('\nRun test_standardize')
#         
#         dataset = self.init_dataset()
#         
#         input_set, _ = self.generate_train_set()
#                 
#         standarized_set = standardize(input_set)
#         
#         assert np.mean(standarized_set[0]) == pytest.approx(0)
#         assert np.std(standarized_set[0]) == pytest.approx(1)
#         
#         print(input_set[0])
#         print(standarized_set[0])        
# =============================================================================
        
        
if __name__ == '__main__':
    unittest.main()        
    
    