import numpy as np
from datetime import datetime
from plotly.graph_objects import Candlestick, Figure
from plotly.offline import plot

from dataset_generation import Dataset

class PredictionPlotter:
    
    
    def __init__(self, model):
        self.model = model
        self.dataset = Dataset(0,0,[]) # dummy values
        
        self.prediction_interval = model.layers[0].input_shape[1]

    
    def do_stuff(self, filename, start=0, end=-1):
        
        csv_list = self.dataset.parse_csv(filename) 
        
        if start < self.prediction_interval:
            start = self.prediction_interval
            
        if end >= len(csv_list):
            end = -1
        elif end < start:
            end = start        
        
        analyzed_interval = csv_list[start - self.prediction_interval:end, 3:9]
        analyzed_interval = np.delete(analyzed_interval, 1, axis=1)

        print(analyzed_interval)

        predictions = np.zeros((end + self.prediction_interval - start), 
                               dtype='int32')

        for i, day in enumerate(analyzed_interval):            
            if i >= self.prediction_interval:
                input_start = i - self.prediction_interval
                input_days = analyzed_interval[input_start:i, 1:5]
                input_days = input_days.astype(dtype='float')
                input_days = np.expand_dims(input_days, axis=0)

                single_prediction = self.model.predict(input_days)
                print(single_prediction)
                predictions[i] = np.argmax(single_prediction)
                
            
        analyzed_interval = analyzed_interval[self.prediction_interval:]
        predictions = predictions[self.prediction_interval:]
    
        print(analyzed_interval)
        print(predictions)
        
        
# =============================================================================
#     def plot(self):
#         
#         open_ = single_input[:,0]
#         high_ = single_input[:,1]
#         low_ = single_input[:,2]
#         close_ = single_input[:,3]
#     
#         dates = list(range(single_input.shape[0]))
#         
#         fig = Figure(data=[Candlestick(x=dates,open=open_, high=high_, low=low_, 
#                                        close=close_)])
#     
#         plot(fig)
# =============================================================================
        