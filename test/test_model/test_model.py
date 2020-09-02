from dataset_generation import Dataset
import predictor


if __name__ == '__main__':
    
    filename = ('../../dataset_generation/datasets_npz/' + 
                    'test_dataset_IN3PRED2CAT[1,2,3].npz')
    dataset = Dataset.load(filename)

    train_input_set, train_label_set = dataset.get_train_set()
    
    model = predictor.model_1(dataset)     
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=256, epochs=1)
    
    model.summary()    
    
    import keras 
    keras.models.save_model(model, 'temp.h5')
    print('-----------------------------------')
    
    
    model_loaded = keras.models.load_model('temp.h5')
    
    model_loaded.summary()
    
    model_loaded.fit(train_input_set, train_label_set, batch_size=256, epochs=1)
    
    