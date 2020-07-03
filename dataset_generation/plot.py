from plotly.graph_objects import Candlestick, Figure
from plotly.offline import plot
from datetime import datetime


def plot_candlestick(single_input):
    
    open_ = single_input[:,0]
    high_ = single_input[:,1]
    low_ = single_input[:,2]
    close_ = single_input[:,3]

    dates = list(range(single_input.shape[0]))
    
    fig = Figure(data=[Candlestick(x=dates,open=open_, high=high_, low=low_, 
                                   close=close_)])
    
    plot(fig)

