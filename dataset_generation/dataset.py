import csv
from copy import deepcopy
import numpy as np

from .misc import LoadingBar


class Dataset:    
    
    def __init__(self, input_interval, prediction_interval, categories):
        self.input_interval = input_interval
        self.prediction_interval = prediction_interval
        self.categories = categories
        
    
    def generate_set(self, filename, set_size=-1):        
        csv_list = self.parse_csv(filename)
        
        input_set, label_set = self.extract_set(csv_list, set_size)        
        
        input_set = self.standardize_input(input_set)
        
        self.count_label_category(label_set)
        
        return input_set, label_set

    def parse_csv(self, filename):
        print('Parsing', filename)
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')        
            csv_list = np.array(list(csv_reader))
            # Delete the first header line
            csv_list = np.delete(csv_list, obj=0, axis=0)
            return csv_list


    def extract_set(self, csv_list, set_size=-1):  
        
        list_size = len(csv_list)  
        max_size = (list_size - self.prediction_interval - self.input_interval 
                    + 1)
        if set_size > max_size or set_size == -1: 
            print('Set size:', max_size)
            set_size = max_size
            start_range = 0
        else:
            start_range = (list_size - self.input_interval - 
                           self.prediction_interval - set_size + 1)
        
        input_set = np.zeros((set_size, self.input_interval, 4), 
                             dtype='float')
        label_set = np.zeros((set_size, len(self.categories) + 1), 
                             dtype='int32')
        
        lb = LoadingBar(size=set_size, message='Generating dataset')
        
        # One iteration samples a next time interval from csv_list
        # If the sample is valid then it is added to the train_set
        end_range = start_range + set_size
        for i, start_day in enumerate(range(start_range, end_range)):
            # start_day - start day of the input interval
            # curr_day - end day of the input interval, 
            #            when the prediction is done 
            curr_day = start_day + self.input_interval - 1
            # pred_end - the last day of prediction_interval, 
            #            until this day the prediction is valid
            pred_end = (start_day + self.input_interval + 
                        self.prediction_interval - 1)
            # If the target interval is not covering a single crypto
            # then reject it and continue
            if (csv_list[start_day][1] != csv_list[pred_end][1]):
                continue
            
            # Slice the start, high, low and close prices from csv_list            
            input_interval = np.array(csv_list[start_day:curr_day + 1, 5:9], 
                                      dtype='float')
            
            # If the standard deviation of the prices in the chosen interval 
            # is zero then reject this example and continue
            if np.std(input_interval) == 0:
                raise Exception('Standard deviation of test set example is 0')  
                continue             

            # Add to training set
            input_set[i] = input_interval            
            label_set[i] = self.calculate_label(curr_day, csv_list)  
           
            lb() # Loading bar update 
        
        return input_set, label_set    
    
    def calculate_label(self, curr_day, csv_list):        
        label = np.zeros((len(self.categories) + 1), dtype='int32') 
        
        # base_price - the current day's close price         
        base_price = highest_price = (float)(csv_list[curr_day][8])
        
        # Search highest price in open and close prices of the prediction
        # interval days
        for i in range(self.prediction_interval):
            highest_price = max(highest_price, 
                                (float)(csv_list[curr_day + 1 + i][5]),
                                (float)(csv_list[curr_day + 1 + i][8]))

        # Ratio of the highest opening or closing prices of the prediciton 
        # interval days and the target day's close price (>1)
        increment_rate = (float)(highest_price) / (float)(base_price) 
        
        self.categories.sort(reverse=True)
        set_category = False
        
        # Find which category the highest price fits in
        for i, c in enumerate(self.categories):
            if increment_rate >= 1 + (c / 100):
                label[len(self.categories) - i] = 1
                set_category = True
                break      
        
        # If no category found, set the base category
        if not set_category:
            label[0] = 1         
            
        return label  
    
    
    def standardize_input(self, input_set):
        """
        Standardize every training example's input separately by moving the
        mean to 0 and scaling the standard deviation to 1
    
        Parameters
        ----------
        input_set : np.ndarray(3D) - 
            DESCRIPTION.
    
        Returns
        -------
        input_set : TYPE
            DESCRIPTION.
    
        """
        print('Standardize dataset') 
        input_set = deepcopy(input_set)    
        for i in range(input_set.shape[0]):   
            if np.std(input_set[i]) == 0:
                print(input_set[i])
            input_set[i] = ((input_set[i] - np.mean(input_set[i])) / 
                            np.std(input_set[i]))
        return input_set

    def count_label_category(self, label_set):        
        sum_label = np.zeros((len(self.categories) + 1), dtype='int32')
        
        for label in label_set:
            sum_label += label
          
        print('Label distribution:', sum_label)
        
        return sum_label        


    def save(self, filename):
        pass