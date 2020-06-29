import unittest

from dataset_generation import generate


class TestGenerate(unittest.TestCase):
    
    def test_generate(self):
        print('test_generate')
        train_set = generate(input_interval=6, 
                             prediction_interval=2, 
                             set_size=1, 
                             categories=[1,2.5,5],
                             train_data='test_data/test_crypto-markets.csv', 
                             test_data='test_data/test_crypto-markets.csv')
        
        input_set, label_set = train_set
        
        assert input_set[0][4][0] == 116.38
        assert label_set[0][2] == 0
        assert label_set[0][3] == 1
        
        print(input_set)
        print(label_set)
        
        
if __name__ == '__main__':
    unittest.main()        