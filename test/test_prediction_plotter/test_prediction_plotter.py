from dataset_generation import Dataset
from model import model_1, model_3
from prediction_plotter import PredictionPlotter


if __name__ == '__main__':
    
    filename = ('../../dataset_generation/datasets_npz/' + 
                    'test_dataset_IN3PRED2CAT[1,2,3].npz')
    dataset = Dataset.load(filename)

    train_input_set, train_label_set = dataset.get_train_set()
    
    model = model_1(dataset)     
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=256, epochs=1)
    
    filename='../../dataset_generation/data_csv/crypto-markets_ONLY_BTC.csv'
    PredictionPlotter(model, filename, start=4, end=16)
    
    
    
    