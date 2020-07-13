from dataset_generation import DatasetManager


"""
This script is used to generate datasets.
It creates a Dataset instance, generates training and test sets and
saves them into 'dataset_generation/datasets' folder
"""
if __name__ == '__main__':
    
    input_interval=60
    prediction_interval=2
    categories=[200]    
    
    data_man = DatasetManager(input_interval, prediction_interval, categories)
    
    train_data = '../dataset_generation/data/crypto-markets_ONLY_BTC.csv'
    train_set = data_man.generate_set(filename=train_data)
    
    test_data = '../dataset_generation/data/crypto-markets_ONLY_BTC.csv'
    test_set = data_man.generate_set(filename=test_data)
    
    categories.sort()    
    dataset_name = ('../dataset_generation/datasets/IN' + 
                    str(input_interval) + 'PRED' + str(prediction_interval) + 
                    'CAT' + str(categories).replace(' ', '') + '.npz')
    
    DatasetManager.save(dataset_name, train_set, test_set)
    


