##### Time Series with TensorFlow for learning purposes by Josh Cooper and Parker Baker #####

### Utilize "Requirements.txt" to get the correct dependencies for the packages below

## Importing in packages
import numpy as np
import pandas as pd
from datetime import datetime
import os
import IPython
import IPython.display
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

## Load in the data

# Replace with your own path
path = r"C:\Users\parke\OneDrive\Documents\MSA\Fall II\TimeSeriesHW2\Data"
# Datasets (proprietary)
energy = pd.read_csv(path + "\hrl_load_metered.csv", index_col=[0], parse_dates=[0])
energy_t1 = pd.read_csv(
    path + "\hrl_load_metered - test1.csv", index_col=[0], parse_dates=[0]
)
energy_t2 = pd.read_csv(
    path + "\hrl_load_metered - test2.csv", index_col=[0], parse_dates=[0]
)
energy_t3 = pd.read_csv(
    path + "\hrl_load_metered - test3.csv", index_col=[1], parse_dates=[1]
)
energy_t4 = pd.read_csv(
    path + "\hrl_load_metered - test4.csv", index_col=[0], parse_dates=[0]
)
energy_t5 = pd.read_csv(
    path + "\hrl_load_metered - test5.csv", index_col=[0], parse_dates=[0]
)
