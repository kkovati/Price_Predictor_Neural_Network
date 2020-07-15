from dataset_generation import Dataset
import predictor


if __name__ == '__main__':
    
    filename = ('../../dataset_generation/datasets_npz/' + 
                    'test_dataset_IN3PRED2CAT[1,2,3].npz')
    dataset = Dataset.load(filename)

    train_input_set, train_label_set = dataset.get_train_set()
    
    model = predictor.model_6(dataset)     
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=256, epochs=1)
    
    model.summary()
    
    

    from keras import backend as K
    import numpy as np
    
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))