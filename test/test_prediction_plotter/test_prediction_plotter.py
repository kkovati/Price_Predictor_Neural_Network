from keras.layers import Input
from keras.models import Model


from dataset_generation import Dataset
from model import model_1
from prediction_plotter import PredictionPlotter


if __name__ == '__main__':
    
    dataset_name = ('../../dataset_generation/datasets/' + 
                    'test_dataset_IN60CAT4.npz')
    train_set, test_set = Dataset.load(dataset_name)
    train_input_set, train_label_set = train_set
    
    model = model_1(4)  
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=128, epochs=1, 
              verbose=0)
        
    assert model.layers[0].input_shape[1] == 60
    assert model.layers[0].input_shape[2] == 4
    
    print(model.layers[0].input_shape)