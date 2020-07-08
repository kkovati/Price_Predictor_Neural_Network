from dataset_generation import Dataset
#from model import model_1, model_2



input_interval=30
prediction_interval=2                     
categories=[1,2.5,5]
category_count = len(categories) + 1


dataset = Dataset(input_interval, prediction_interval, categories)

train_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
train_set = dataset.generate_set(filename=train_data)
train_input_set, train_label_set = train_set

print(train_input_set[0])

dataset.save('asdasd.npz', train_input_set, train_label_set)

train_input_set, train_label_set = dataset.load('asdasd.npz')

print(train_input_set[0])

# =============================================================================
# models = [model_1(category_count),
#           model_2(category_count)]
# 
# 
# for model in models:
#     model.compile(optimizer='adam', loss='categorical_crossentropy', 
#                   metrics=['accuracy'])
#     print('\nTrain', model.name)
#     model.fit(train_input_set, train_label_set, batch_size=128, epochs=5)
#     #model.summary()
# 
# 
# 
# test_data = 'dataset_generation/data/crypto-markets_ONLY_BTC.csv'
# test_set = dataset.generate_set(filename=test_data)       
# test_input_set, test_label_set = test_set
# 
# for model in models:
#     print('\nTest', model.name)
#     loss = model.evaluate(test_input_set, test_label_set, batch_size=128)
#     print(loss)
# =============================================================================


