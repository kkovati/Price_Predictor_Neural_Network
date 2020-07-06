from dataset_generation import Dataset
#from model import cnn_model


 
dataset = Dataset(input_interval=6, 
                  prediction_interval=2,                      
                  categories=[1,2.5,5])

train_data = 'dataset_generation/data/crypto-markets_NO_BTC.csv'
train_set = dataset.generate_training_set(set_size=1000, data=train_data)

test_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
test_set = dataset.generate_training_set(set_size=100, data=test_data)

       
input_set, label_set = train_set

print(input_set[1])
print(label_set[1])


#cnn_model((6,4))
