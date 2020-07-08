from keras.layers import Input
from keras.models import Model


from dataset_generation import Dataset
from model import model_1
from prediction_plotter import PredictionPlotter


if __name__ == '__main__':
    
    dataset_name = ('../../dataset_generation/datasets/' + 
                    'test_dataset_IN3PRED2CAT[1,2.5,5].npz')
    train_set, test_set = Dataset.load(dataset_name)
    train_input_set, train_label_set = train_set
    
    model = model_1(4) 
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=256, epochs=1, 
              verbose=0)
    
    pred_plot = PredictionPlotter(model)
    
    pred_plot.do_stuff(filename='../../dataset_generation/data/crypto-markets_ONLY_BTC.csv', start=4, end=6)
    