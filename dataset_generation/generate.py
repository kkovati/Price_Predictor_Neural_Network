from dataset_generation import Dataset


"""
This script creates a Dataset instance, generates training and test sets and
saves them into 'dataset_generation/datasets' folder
"""
if __name__ == '__main__':
    
    input_interval=3
    prediction_interval=2
    categories=[1,2.5,5]    
    
    dataset = Dataset(input_interval, prediction_interval, categories)
    
    train_data = '../dataset_generation/data/crypto-markets_ONLY_BTC.csv'
    train_set = dataset.generate_set(filename=train_data)
    
    test_data = '../dataset_generation/data/crypto-markets_ONLY_BTC.csv'
    test_set = dataset.generate_set(filename=test_data)
    
    categories.sort()    
    dataset_name = ('../dataset_generation/datasets/IN' + 
                    str(input_interval) + 'PRED' + str(prediction_interval) + 
                    'CAT' + str(categories).replace(' ', '') + '.npz')
    
    dataset.save(dataset_name, train_set, test_set)
    


