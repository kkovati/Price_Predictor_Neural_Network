from dataset_generation import generate


train_set = generate(input_interval=6, 
                     prediction_interval=2,                      
                     categories=[1,2.5,5],
                     training_set_size=100000, 
                     test_set_size=100,
                     train_data='dataset_generation/data/crypto-markets_NO_BTC.csv', 
                     test_data='dataset_generation/data/crypto-markets_ONLY_BTC.csv')
        
input_set, label_set = train_set

print(input_set.shape)
print(label_set.shape)