import unittest
import pytest
import numpy as np

from dataset_generation import generate, plot_candlestick



class TestPlot(unittest.TestCase):
    
    def generate_train_set(self):
        return generate(input_interval=6, 
                        prediction_interval=2, 
                        categories=[1,2.5,5],
                        training_set_size=10, 
                        test_set_size=1,
                        train_data='test_data/test_crypto-markets.csv', 
                        test_data='test_data/test_crypto-markets.csv')
    
    
    def test_plot_candlestick(self):
        print('\nRun test_plot')        
        
        input_set, _ = self.generate_train_set()
        
        plot_candlestick(input_set[0])
        
        print(input_set[0])
        
         
if __name__ == '__main__':
    unittest.main()        