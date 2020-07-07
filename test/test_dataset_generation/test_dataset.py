import unittest
import pytest
import numpy as np

from dataset_generation import Dataset


class TestDataset(unittest.TestCase):
    
    def setUp(self):
        self.dataset = Dataset(input_interval=6, 
                               prediction_interval=2, 
                               categories=[1,2.5,5])
    
    def test_parse_csv(self):        
        print('\n---Run test_parse_csv---')
            
        filename = '../test_data/test_crypto-markets_8_lines.csv'
        csv_list = self.dataset.parse_csv(filename)
        
        print(csv_list)
    
    def test_extract_set(self):
        print('\n---Run test_generate_training_set---')
        
        filename = '../test_data/test_crypto-markets_9_lines.csv'
        csv_list = self.dataset.parse_csv(filename)
        
        training_set = self.dataset.extract_set(csv_list)

        input_set, _ = training_set
        
        assert input_set[0][4][0] == 116.38
        assert label_set[0][2] == 0
        assert label_set[0][3] == 1
        
        print(input_set[0])
        print(label_set[0])
        print(input_set[1])        
        print(label_set[1])
        
    def test_calculate_label(self):
        print('\n---Run test_calculate_label---')
        
        filename = '../test_data/test_crypto-markets_9_lines.csv'
        csv_list = self.dataset.parse_csv(filename)
        
        input_set, label_set = test_set
        
        assert label_set[0][2] == 0
        assert label_set[0][3] == 1
        
        print(input_set[0])
        print(label_set[0])
        print(input_set[1])        
        print(label_set[1])
        
            
    def test_standardize_input(self):
        print('\nRun test_standardize_input')
        
        filename = '../test_data/test_crypto-markets_9_lines.csv'
        csv_list = self.dataset.parse_csv(filename)
        
        input_set, label_set = test_set
                
        standarized_input_set = standardize(input_set)
        
        assert np.mean(standarized_set[0]) == pytest.approx(0)
        assert np.std(standarized_set[0]) == pytest.approx(1)
        
        print(input_set[0])
        print(standarized_set[0])        
        
        
if __name__ == '__main__':
    unittest.main()        
    
    