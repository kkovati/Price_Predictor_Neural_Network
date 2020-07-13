from dataset_generation import DatasetManager
from model import model_1, model_2
from prediction_plotter import PredictionPlotter



"""
This script ....
"""
if __name__ == '__main__':

    categories = [2] 
    dataset_name = 'dataset_generation/datasets/IN60PRED2CAT[2].npz'
    train_set, test_set = DatasetManager.load(dataset_name)
    
    train_input_set, train_label_set = train_set 
    test_input_set, test_label_set = test_set
    
    model = model_1(len(categories) + 1)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=128, epochs=1,
              shuffle=True)    
    
    print('Test results:')
    loss = model.evaluate(test_input_set, test_label_set)
    print(loss)
    
    filename='dataset_generation/data/crypto-markets_ONLY_BTC.csv'
    PredictionPlotter(model, filename, categories, start=4, end=16)




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


