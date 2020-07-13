from dataset_generation import DatasetManager
from model import model_1
from prediction_plotter import PredictionPlotter


if __name__ == '__main__':
    
    categories = [1,2.5,5]
    
    dataset_name = ('../../dataset_generation/datasets/' + 
                    'test_dataset_IN3PRED2CAT[1,2.5,5].npz')
    train_set, test_set = DatasetManager.load(dataset_name)

    train_input_set, train_label_set = train_set
    
    model = model_1(4)     
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=256, epochs=1)
    
    filename='../../dataset_generation/data/crypto-markets_ONLY_BTC.csv'
    PredictionPlotter(model, filename, categories, prediction_interval=2, start=4, end=16)
    
    
    
    