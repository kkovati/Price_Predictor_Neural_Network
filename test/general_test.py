from dataset_generation import Dataset
from predictor import model_1, model_2, model_3, model_4, model_5, model_6
from prediction_plotter import PredictionPlotter


"""
This script is used to test a lifecycle of a model
"""
if __name__ == '__main__':

    dataset_name = '../dataset_generation/datasets_npz/IN60PRED1CAT[1,5].npz'
    # dataset_name = '../dataset_generation/datasets_npz/test_dataset_IN3PRED2CAT[1,2,3].npz'
    dataset = Dataset.load(dataset_name)
    
    train_input_set, train_label_set = dataset.get_train_set()
    test_input_set, test_label_set =  dataset.get_test_set()
    
    model = model_1(dataset)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=128, epochs=1,
              shuffle=True)    
    
    print('Test results:')
    loss = model.evaluate(test_input_set, test_label_set)
    print(loss)
    
    filename='dataset_generation/data_csv/crypto-markets_ONLY_BTC.csv'
    PredictionPlotter(model, filename, start=-140, end=-100)


