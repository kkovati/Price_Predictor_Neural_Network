import unittest
import pytest
import numpy as np

from dataset_generation import plot_candlestick
from test_dataset_generation import TestDatasetGeneration


class TestPlot(unittest.TestCase):
    
    def test_plot_candlestick(self):
        print('\nRun test_plot')        
        
        input_set, _ = TestDatasetGeneration().generate_train_set()
        
        plot_candlestick(input_set[0])
        
        print(input_set[0])
        
         
if __name__ == '__main__':
    unittest.main()        