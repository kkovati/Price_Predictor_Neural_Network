from dataset_generation import Dataset
#from model import model_1, model_2



"""
This script ....
"""
if __name__ == '__main__':

    dataset_name = 'dataset_generation/datasets/IN60PRED3CAT[1,2.5,5].npz'

    train_set, test_set = Dataset.load(dataset_name)
    
    print(train_set[0].shape)
    print(train_set[0][0][0])
    print(test_set[0].shape)
    print(test_set[0][0][0])

# SHUFFLE training set


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


