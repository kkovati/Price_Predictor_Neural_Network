from dataset_generation import Dataset
from model import model_1, model_2


 
dataset = Dataset(input_interval=30, 
                  prediction_interval=3,                      
                  categories=[1,2.5,5])

train_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
train_set = dataset.generate_training_set(set_size=1000, data=train_data)

test_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
test_set = dataset.generate_training_set(set_size=1000, data=test_data)

       
input_set, label_set = train_set

model = model_2(4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(input_set, label_set, epochs=5, batch_size=128)

model.summary()