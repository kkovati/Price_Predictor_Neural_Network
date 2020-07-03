from dataset_generation import generate
#from model import cnn_model

train_set = generate(input_interval=6, 
                     prediction_interval=2,                      
                     categories=[1,2.5,5],
                     training_set_size=100, 
                     test_set_size=1000,
                     train_data='dataset_generation/data/crypto-markets_NO_BTC.csv', 
                     #train_data='dataset_generation/data/crypto-markets_ONLY_BTC.csv', 
                     test_data='dataset_generation/data/crypto-markets_ONLY_BTC.csv')
        
input_set, label_set = train_set



#cnn_model((6,4))
