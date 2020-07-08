import numpy as np
from datetime import datetime
from plotly.graph_objects import Candlestick, Figure
from plotly.offline import plot

from dataset_generation import Dataset

class PredictionPlotter:
    
    
    def __init__(self, model, filename, start=0, end=-1):
        self.model = model
                
        prediction_interval = model.layers[0].input_shape[1]
        
        analyzed_interval, predictions = self.make_prediction(model, filename, prediction_interval, start, end)
        
        print(analyzed_interval)
        
        self.plot(analyzed_interval, predictions)

    
    def make_prediction(self, model, filename, prediction_interval, start=0, 
                        end=-1):
        
        dataset = Dataset(0,0,[]) # dummy values for init
        csv_list = dataset.parse_csv(filename) 
        
        if start < prediction_interval:
            start = prediction_interval
            
        if end >= len(csv_list):
            end = -1
        elif end < start:
            end = start        
        
        analyzed_interval = csv_list[start - prediction_interval:end, 3:9]
        analyzed_interval = np.delete(analyzed_interval, 1, axis=1)

        predictions = np.zeros((end + prediction_interval - start), 
                               dtype='int32')

        for i, _ in enumerate(analyzed_interval):            
            if i >= prediction_interval:
                input_start = i - prediction_interval
                input_example = analyzed_interval[input_start:i, 1:5]
                input_example = input_example.astype(dtype='float')
                input_example = np.expand_dims(input_example, axis=0)
                
                input_example = ((input_example - np.mean(input_example)) / 
                                 np.std(input_example))
                
                single_prediction = model.predict(input_example)
                predictions[i] = np.argmax(single_prediction)                
            
        analyzed_interval = analyzed_interval[prediction_interval:]
        predictions = predictions[prediction_interval:]
    
        return analyzed_interval, predictions
        
        
    def plot(self, analyzed_interval, predictions):
        
        dates = analyzed_interval[:,0]        
        open_ = analyzed_interval[:,1]
        high_ = analyzed_interval[:,2]
        low_ = analyzed_interval[:,3]
        close_ = analyzed_interval[:,4]
    
        
        
        fig = Figure(data=[Candlestick(x=dates, open=open_, high=high_, low=low_, 
                                       close=close_)])
        
        fig.update_layout(title='The Great Recession',
                          yaxis_title='AAPL Stock',
                          shapes = [dict(x0='2013-05-09', x1='2013-05-09', y0=0, y1=1, xref='x', yref='paper',line_width=2)],
                          annotations=[dict(x='2013-05-09', y=0.05, xref='x', yref='paper',showarrow=False, xanchor='left', text='Increase Period Begins')])
        
        fig.add_scatter(y=[2, 3.5, 4], mode="markers",
                        marker=dict(size=20, color="MediumPurple"),
                        name="c", row=1, col=2)
    
        plot(fig)
        