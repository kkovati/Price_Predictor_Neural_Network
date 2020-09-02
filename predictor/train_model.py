import keras

from dataset_generation import Dataset
from predictor import model_1, model_2, model_3, model_4, model_5, model_6 


"""
This script is used to train, test and save models
It loads a dataset, calls a model generator functions, compiles, trains and
tests the model then saves it into the 'predictor/trained_models_h5' folder
"""
if __name__ == '__main__':
    
    #dataset_name = '../dataset_generation/datasets_npz/IN60PRED1CAT[1,5].npz'
    dataset_name = '../dataset_generation/datasets_npz/test_dataset_IN3PRED2CAT[1,2,3].npz'
    dataset = Dataset.load(dataset_name)
    
    train_input_set, train_label_set = dataset.get_train_set()
    test_input_set, test_label_set =  dataset.get_test_set()
    
    models = [model_4]
    
    for model_generator in models:    
        model = model_generator(dataset)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy']) 
        
        print('---------------------------------------------')
        print('Training of', model.name)
        model.fit(train_input_set, train_label_set, batch_size=128, epochs=1,
                  shuffle=True)    
        
        print('Test results:')

        loss = model.evaluate(test_input_set, test_label_set)
        print(model.metrics_names)
        print(loss)   
        
        path = 'trained_models_h5/'
        model_name = (model.name + '_IN' + str(model.input_interval) + 'PRED' + 
                str(model.prediction_interval) + 'CAT' + 
                str(model.categories).replace(' ', '') + '.h5')
        print('Save', path + model_name)
        model.save(path + model_name)
    
