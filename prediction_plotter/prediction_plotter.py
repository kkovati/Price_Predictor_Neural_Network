import numpy as np
from datetime import datetime
from plotly.graph_objects import Candlestick, Figure
from plotly.offline import plot

from dataset_generation import Dataset

class PredictionPlotter:
    
    
    def __init__(self, model, filename, start=0, end=-1):
        self.model = model
                
        prediction_interval = model.layers[0].input_shape[1]
        
        analyzed_interval, predictions = self.make_predictions(model, filename, prediction_interval, start, end)
        
        
        
        print(analyzed_interval)
        
        self.plot(analyzed_interval, predictions, prediction_interval)

    
    def make_predictions(self, model, filename, prediction_interval, start=0, 
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
                # normalize 
                input_example = ((input_example - np.mean(input_example)) / 
                                 np.std(input_example))
                
                single_prediction = model.predict(input_example)
                predictions[i] = np.argmax(single_prediction)                
            
        analyzed_interval = analyzed_interval[prediction_interval:]
        predictions = predictions[prediction_interval:]
    
        return analyzed_interval, predictions
        
        
    def plot(self, analyzed_interval, predictions, prediction_interval):
    
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout
    # https://plotly.com/python/candlestick-charts/
    # https://plotly.com/python/shapes/
    
        candlestick = Candlestick(x=analyzed_interval[:,0], 
                                  open=analyzed_interval[:,1], 
                                  high=analyzed_interval[:,2], 
                                  low=analyzed_interval[:,3], 
                                  close=analyzed_interval[:,4])
        
        fig = Figure(data=[candlestick])
        
        fig.update_layout(title='The Great Recession',
                          yaxis_title='AAPL Stock',
                          shapes = [dict(x0='2013-05-09', x1='2013-05-09', y0=0, y1=1, xref='x', yref='paper',line_width=2)],
                          annotations=[dict(x='2013-05-09', y=0.05, xref='x', yref='paper',showarrow=False, xanchor='left', text='Increase Period Begins')])
        
        
        
        
        fig.add_shape(type="line",x0='2013-05-03',y0=110, x1='2013-05-04',y1=120,
                      line=dict(color="MediumPurple",width=4,dash="dot",))
        
        fig.add_shape(type="line",x0='2013-05-05',y0=110, x1='2013-05-06',y1=120,
                      line=dict(color="Red",width=4,dash="dot",))


        self.visualize_predictions(fig, analyzed_interval, predictions, 
                                   prediction_interval)
       
    
        plot(fig)
        
    def visualize_predictions(self, figure, analyzed_interval, predictions, 
                              prediction_interval):
        
        for i, prediction in enumerate(predictions):
            if i < len(analyzed_interval) - prediction_interval:
                figure.add_shape(type="line",
                                 x0=analyzed_interval[i][0],
                                 y0=analyzed_interval[i][4], # Close price
                                 x1=analyzed_interval[i + prediction_interval][0],
                                 y1=(float)(analyzed_interval[i][4]) * 1.1,
                                 line=dict(color="Red",width=4, dash="dot"))
        
        