import csv
import numpy as np
from copy import deepcopy

from .misc import LoadingBar

# https://www.kaggle.com/jessevent/all-crypto-currencies

def generate(input_interval, prediction_interval, categories, 
             training_set_size, test_set_size, train_data, test_data, 
             save=False, filename=''):
    
    training_set = generate_training_set(input_interval,prediction_interval, 
                                         categories, training_set_size, 
                                         train_data)
    
    if save:
        pass
    
    return training_set
    

def generate_training_set(input_interval, prediction_interval, categories, 
                          set_size, train_data):
    
    input_set = np.zeros((set_size, input_interval, 4), dtype='float')
    label_set = np.zeros((set_size, len(categories) + 1), dtype='int32') 
    
    print('Parsing data file...')
    with open(train_data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')        
        csv_list = list(csv_reader) # close the file here
       
    size = len(csv_list)    
    index = 0
    
    lb = LoadingBar(size=set_size, message='Generating training set')
    
    # one iteration of the while loop samples from the price timeline of a 
    # random cryptocurrency and random time interval
    # after checking the date is added to the training set list
    while(index < set_size):
        target_day = np.random.randint(input_interval, 
                                       size - prediction_interval)
        target_crypto = csv_list[target_day][1]
        if (target_crypto != csv_list[target_day - input_interval + 1][1] or
            target_crypto != csv_list[target_day + prediction_interval][1]):
            continue
        
        for i in range(input_interval):
            for j in range(4):
                input_set[index][i][j] = csv_list[target_day - input_interval + 1 + i][j + 5]            
        
        base_price = highest_price = (float)(csv_list[target_day][8]) # target day close
        
        for i in range(prediction_interval):

            highest_price = max(highest_price, (float)(csv_list[target_day + 1 + i][5])) # open
            highest_price = max(highest_price, (float)(csv_list[target_day + 1 + i][8])) # close
            
        increment_rate = (float)(highest_price) / (float)(base_price) # >1
        
        categories.sort(reverse=True)
        flag = False
        
        for i, c in enumerate(categories):
            if increment_rate >= 1 + (c / 100):
                label_set[index][len(categories) - i] = 1
                flag = True
                break      
        
        if not flag:
            label_set[index][0] = 1            
            
        index += 1           
        lb()    
    
    return input_set, label_set


def generate_test_set(input_interval, prediction_interval, set_size,  
                      train_data, test_data,):
    pass


def standardize(input_set):
    
    input_set = deepcopy(input_set)
    
    for i in range(input_set.shape[0]):        
        input_set[i] = (input_set[i] - np.mean(input_set[i])) / np.std(input_set[i])
            
    return input_set

