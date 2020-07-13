import numpy as np
from plotly.graph_objects import Candlestick, Figure
from plotly.offline import plot

from dataset_generation import Dataset

class PredictionPlotter:
    
    
    def __init__(self, model, filename, start=0, end=-1):
        
        self.model = model
                
        analyzed_interval, predictions = self.make_predictions(model, filename, 
                                                               start, end)    
        
        figure = self.init_figure(analyzed_interval, predictions, model)
        
        self.visualize_predictions(figure, analyzed_interval, predictions, 
                                   model)
       
        plot(figure)        

    
    def make_predictions(self, model, filename, start=0, end=-1):
        
        dm = Dataset(0,0,[]) # dummy values for init
        csv_list = dm.parse_csv(filename) 
        
        input_interval = model.input_interval        
        if start < input_interval and start > 0:
            start = input_interval
            
        if end >= len(csv_list):
            end = -1
        elif end < start:
            end = start        
        
        # slice date and prices from csv_list
        analyzed_interval = csv_list[start - input_interval:end, 3:9]
        # delete ranknow column
        analyzed_interval = np.delete(analyzed_interval, 1, axis=1)
        
        # prediction is a scalar now, representing the hardmax output index 
        # of the model
        predictions = np.zeros((end + input_interval - start), 
                               dtype='int32')

        for i, _ in enumerate(analyzed_interval):            
            if i >= input_interval:
                input_start = i - input_interval
                input_example = analyzed_interval[input_start:i, 1:5]
                input_example = input_example.astype(dtype='float')
                # expand to a singe example batch
                input_example = np.expand_dims(input_example, axis=0)
                # normalize 
                input_example = ((input_example - np.mean(input_example)) / 
                                 np.std(input_example))
                # make prediction
                single_prediction = model.predict(input_example)
                # from softmax output create a hardmax index
                predictions[i] = np.argmax(single_prediction)                
            
        analyzed_interval = analyzed_interval[input_interval:]
        predictions = predictions[input_interval:]
    
        return analyzed_interval, predictions
        
        
    def init_figure(self, analyzed_interval, predictions, model):
    
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout
    # https://plotly.com/python/candlestick-charts/
    # https://plotly.com/python/shapes/
    
        print(analyzed_interval[:,0])
    
        candlestick = Candlestick(x=analyzed_interval[:,0], 
                                  open=analyzed_interval[:,1], 
                                  high=analyzed_interval[:,2], 
                                  low=analyzed_interval[:,3], 
                                  close=analyzed_interval[:,4])
        
        figure = Figure(data=[candlestick])
        
        title = ('Predictions for the next ' + str(model.prediction_interval) + 
                 ' days')
        figure.update_layout(title=title, yaxis_title='Price')
        
        return figure        
        
        
    def visualize_predictions(self, figure, analyzed_interval, predictions, 
                              model):
        
        prediction_interval = model.prediction_interval
        categories = model.categories
        categories.sort()
        
        for i, prediction in enumerate(predictions):            
            # do not draw those predictions which cannot be verified
            if i >= len(analyzed_interval) - prediction_interval:
                return
            
            if prediction == 0:
                ratio = 1
                text = '< 0%' 
                color = 'Red'
            else:                
                ratio = categories[prediction-1] / 100 + 1
                text = '> ' + str(categories[prediction-1]) + '%'
                color = 'Green'
            
            # vertical line
            figure.add_shape(type="line",
                             x0=analyzed_interval[i][0], # date
                             y0=analyzed_interval[i][4], # close price
                             x1=analyzed_interval[i][0],
                             y1=(float)(analyzed_interval[i][4]) * ratio,
                             line=dict(color=color,width=2, dash="dot"))
            
            # horizontal line
            figure.add_shape(type="line",
                             x0=analyzed_interval[i][0],
                             y0=(float)(analyzed_interval[i][4]) * ratio,
                             x1=analyzed_interval[i + prediction_interval][0],
                             y1=(float)(analyzed_interval[i][4]) * ratio,
                             line=dict(color=color,width=2, dash="dot"))
            
            figure.add_annotation(x=analyzed_interval[i][0],
                                  y=(float)(analyzed_interval[i][4]) * ratio,
                                  text=text)
        
        