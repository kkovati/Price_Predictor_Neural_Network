from dataset_generation import Dataset
import model


if __name__ == '__main__':
    
    filename = ('../../dataset_generation/datasets_npz/' + 
                    'test_dataset_IN3PRED2CAT[1,2,3].npz')
    dataset = Dataset.load(filename)

    train_input_set, train_label_set = dataset.get_train_set()
    
    model = model.model_3(dataset)     
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(train_input_set, train_label_set, batch_size=256, epochs=1)
    
    model.summary()
    