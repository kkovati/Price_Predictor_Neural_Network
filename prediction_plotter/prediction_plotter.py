from datetime import datetime
from plotly.graph_objects import Candlestick, Figure
from plotly.offline import plot

from dataset_generation import Dataset

class PredictionPlotter:
    
    
    def __init__(self, model):
        self.model = model
        self.dataset = Dataset(0,0,0) # dummy values
            
    
    def do_stuff(self, start_date, end_date='2018-11-29', filename):
        
        csv_list = self.dataset.parse_csv(filename) 
        
        for
    
    def plot(self):
        
        open_ = single_input[:,0]
        high_ = single_input[:,1]
        low_ = single_input[:,2]
        close_ = single_input[:,3]
    
        dates = list(range(single_input.shape[0]))
        
        fig = Figure(data=[Candlestick(x=dates,open=open_, high=high_, low=low_, 
                                       close=close_)])
    
        plot(fig)
        