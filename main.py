from dataset_generation import Dataset
from model import model_1, model_2


 
dataset = Dataset(input_interval=30, 
                  prediction_interval=2,                      
                  categories=[1,2.5,5])

train_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
train_set = dataset.generate_set(filename=train_data)

test_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
test_set = dataset.generate_test_set(set_size=1000, data=test_data)

       
train_input_set, train_label_set = train_set

model = model_2(4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_input_set, train_label_set, batch_size=128, epochs=5)



model.summary()

test_input_set, test_label_set = test_set

loss = model.evaluate(test_input_set, test_label_set, batch_size=128)

print(loss)