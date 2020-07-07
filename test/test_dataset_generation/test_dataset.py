import unittest
import pytest
import numpy as np

from dataset_generation import Dataset


class TestDataset(unittest.TestCase):
    
    def setUp(self):
        self.dataset = Dataset(input_interval=6, 
                               prediction_interval=2, 
                               categories=[1,2.5,5])
        
        self.filename = '../test_data/test_crypto-markets_9_lines.csv'
        
    
    def test_parse_csv(self):        
        print('\n---Run test_parse_csv---')
            
        csv_list = self.dataset.parse_csv(self.filename)
        
        assert float(csv_list[1][5]) == 134.44
        
        print(csv_list)
    
    
    def test_extract_set(self):
        print('\n---Run test_generate_training_set---')
        
        csv_list = self.dataset.parse_csv(self.filename)        
        training_set = self.dataset.extract_set(csv_list)

        input_set, _ = training_set
        
        assert input_set[0][4][0] == 116.38
        assert input_set[1][4][3] == 97.75
        
        print(input_set[0])
        print(input_set[1])        

        
    def test_calculate_label(self):
        print('\n---Run test_calculate_label---')
        
        csv_list = self.dataset.parse_csv(self.filename)
        
        training_set = self.dataset.extract_set(csv_list)        
        _, label_set = training_set
        
        # 115.91 / 97.75 == 1.18 -> 18% > 5% 
        assert all(label_set[0] == [0,0,0,1])

        
        # 115.98 / 112.5 == 1.03 -> 5% > 3% > 2.5%
        assert all(label_set[1] == [0,0,1,0])
        
        print(label_set[0])
        print(label_set[1])
        
            
    def test_standardize_input(self):
        print('\nRun test_standardize_input')
        
        csv_list = self.dataset.parse_csv(self.filename)
        
        training_set = self.dataset.extract_set(csv_list)        
        input_set, _ = training_set
                
        standarized_input_set = self.dataset.standardize_input(input_set)
        
        assert np.mean(standarized_input_set[0]) == pytest.approx(0)
        assert np.std(standarized_input_set[0]) == pytest.approx(1)
        
        print(input_set[0])
        print(standarized_input_set[0])

        
    def test_count_label_category(self):
        print('\n---Run test_count_label_category---')
        
        csv_list = self.dataset.parse_csv(self.filename)
        
        training_set = self.dataset.extract_set(csv_list)        
        _, label_set = training_set
        
        sum_label = self.dataset.count_label_category(label_set)
        
        assert all(sum_label == [0,0,1,1])
        
        print(sum_label)
        
        
if __name__ == '__main__':
    unittest.main()        
    
    