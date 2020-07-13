from dataset_generation import Dataset


"""
This script is used to generate datasets
It creates a Dataset instance, generates training and test sets and
saves them into 'dataset_generation/datasets' folder
"""
if __name__ == '__main__':
    
    input_interval=60
    prediction_interval=2
    categories=[1,2,3]   
    
    train_data = '../dataset_generation/data_csv/crypto-markets_NO_BTC.csv'
    test_data = '../dataset_generation/data_csv/crypto-markets_ONLY_BTC.csv'
    
    dataset = Dataset(input_interval, prediction_interval, categories)
    test_set = dataset.generate(train_data, test_data)
    
    categories.sort()    
    dataset_name = ('../dataset_generation/datasets_npz/IN' + 
                    str(input_interval) + 'PRED' + str(prediction_interval) + 
                    'CAT' + str(categories).replace(' ', '') + '.npz')
    
    dataset.save(dataset_name)
    


