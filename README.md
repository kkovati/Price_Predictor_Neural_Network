# Price Predictor Neural Network

Deep learning neural network predicts cryptocurrency prices. The neural network uses special layers:
combination of layers used in CNNs (Convolutional Neural Network) and RNNs (Recurrent NN).  

**This project is under development!**

### Model

Multiple models are trained and tested concurrently to find an optimal model structure for the problem.
The models are using special Inception and ConvGRU layers described below.<br/>
Source:<br/>
https://github.com/kkovati/Price_Predictor_Neural_Network/blob/master/predictor/models

**Inception Layer**

An inception layer is made up multiple different layers concatenated together. 
In this implementation the inception layer built up multiple different kernel size 1D convolutional layers.
The different sized kernels find different size patterns in the one dimensional time series data.
(This time series can have multiple channels, e.g. open, close, low and high prices)<br/>
Source:<br/>
https://github.com/kkovati/Price_Predictor_Neural_Network/blob/master/predictor/layers/inception_layer.py

**ConvGRU layer**

A ConvGRU layer is a combination of the previously described Inception layer and a GRU (Gated recurrent unit) layer
concatenated together.
A GRU layer processes the data time dependent, in contrast to the convolutional layer,
so the two different type of layers finds different type of patterns.<br/>
Source:<br/>
https://github.com/kkovati/Price_Predictor_Neural_Network/blob/master/predictor/layers/conv_gru_layer.py

### Dataset

For training and testing the following dataset is used:<br/>
https://www.kaggle.com/jessevent/all-crypto-currencies

This dataset contains nearly 1 million records of daily open, close, low and high prices
for more than 2000 cryptocurrencies.

The dataset management is done in a single class which is responsible for 
reading, preprocessing and transforming the raw data.
Also it can save and load the dataset in a compressed .npz format.<br/>
Source:<br/>
https://github.com/kkovati/Price_Predictor_Neural_Network/blob/master/dataset_generation/dataset.py 

### Prediction Results

The prediction is visualized in a price chart with Plotly. 
This is a sample prediciton output in an interactive HTML format:<br/>
https://htmlpreview.github.io/?https://github.com/kkovati/Price_Predictor_Neural_Network/blob/master/prediction_plotter/prediction_plots/model_1_prediction_sample.html

A printscreen of this chart:

![Prediction_Results](https://github.com/kkovati/Price_Predictor_Neural_Network/blob/master/documentation/model_1_predictions_sample.png?raw=true)





