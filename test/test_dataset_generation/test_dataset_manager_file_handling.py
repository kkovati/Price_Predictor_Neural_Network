import unittest


from dataset_generation import DatasetManager


class TestDataset(unittest.TestCase):
    
    def setUp(self):
        data_man = DatasetManager(input_interval=6, 
                                  prediction_interval=2, 
                                  categories=[1,2.5,5])
        
        filename = '../test_data/test_crypto-markets_9_lines.csv'
        
        self.in_set_0, self.lab_set_0 = data_man.generate_set(filename)
        self.in_set_1, self.lab_set_1 = data_man.generate_set(filename)        
        
    def test_save(self):
        print('\n---Run test_...---')
        
        
        
        pass
        
        
if __name__ == '__main__':
    unittest.main()  