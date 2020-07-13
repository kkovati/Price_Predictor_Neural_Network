import unittest

from dataset_generation import Dataset


class TestDataset(unittest.TestCase):
    
    def setUp(self):
        
        self.input_interval = 6
        self.prediction_interval = 2 
        self.categories = [1,2.5,5]
        
        self.dataset = Dataset(self.input_interval, 
                               self.prediction_interval, 
                               self.categories)
        
        datafile = '../test_data/test_crypto-markets_9_lines.csv'        
        self.dataset.generate(datafile, datafile)
               
        self.filename = 'test_dataset.npz'        
        self.dataset.save(self.filename)        
        
    
    def test_load(self):
        print('\n---Run test_load---')
        
        dataset = Dataset.load(self.filename)
        
        dataset.get_test_set()
        
        
if __name__ == '__main__':
    unittest.main()  