import unittest
import pytest
import numpy as np

from dataset_generation import Dataset


class TestPlot(unittest.TestCase):
    
    def setUp(self):
        print('setUp')
        self.dataset = Dataset(input_interval=6, 
                               prediction_interval=3, 
                               categories=[1, 2.5, 5])
    
    
    def test_plot_candlestick(self):
        print('\nRun test_plot')   
        
        train_data = '../test_data/test_crypto-markets_9_lines.csv'
        input_set, _ = self.dataset.generate_training_set(set_size=1000, 
                                                          data=train_data)
        
        self.dataset.plot_candlestick(input_set[0])
        
        print(input_set[0])
        
         
if __name__ == '__main__':
    unittest.main()        