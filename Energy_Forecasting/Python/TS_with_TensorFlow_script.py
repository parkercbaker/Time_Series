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

### Preprocessing and cleanup
# Combine datasets
total_energy = pd.concat(
    [energy, energy_t1, energy_t2, energy_t3, energy_t4, energy_t5]
)
# Drop unnessesary columns
aep_df = total_energy.drop(
    columns=[
        "datetime_beginning_utc",
        "nerc_region",
        "zone",
        "mkt_region",
        "load_area",
        "is_verified",
    ]
)
# Rename Index
aep_df.index.names = ["Datetime"]
# Sort Data
aep_df.sort_index(inplace=True)
# Identify Duplicate Indices
duplicate_index = aep_df[aep_df.index.duplicated()]
print(aep_df.loc[duplicate_index.index.values, :])
# Replace Duplicates with Mean Value
aep_df = aep_df.groupby("Datetime").agg(np.mean)
# Set Datetime Index Frequency
aep_df = aep_df.asfreq("H")
# Determine # of Missing Values
print("# of Missing df_MW Values:{}".format(len(aep_df[aep_df["mw"].isna()])))
# Impute Missing Values
aep_df["mw"] = aep_df["mw"].interpolate(limit_area="inside", limit=None)
print(
    "After interpolating - # of Missing df_MW Values:{}".format(
        len(aep_df[aep_df["mw"].isna()])
    )
)

## Feature Engineering -- Creation of cyclic features
# Creates a datetime object
date_time = pd.to_datetime(aep_df.index, format="%d.%m.%Y %H:%M:%S")
# Converts to Unix time
timestamp_s = date_time.map(pd.Timestamp.timestamp)
# Creation of cyclic features to be used in Fourier Transforms later (periodic patterns within data)
day = 24 * 60 * 60
year = (365.2425) * day
# Columns for cyclic features of both day and year
aep_df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
aep_df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
aep_df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
aep_df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

## Fourier Transformations

# Creates a real valued fourier transform of the data which represents the amplitude and phases of the freq components
fft = tf.signal.rfft(aep_df["mw"])

# Creating a numpy array to calculate frequencies within a year
f_per_dataset = np.arange(0, len(fft))  ## **Don't really understand why you do this?
n_samples_h = len(aep_df["mw"])
hours_per_year = (
    24 * 365.2524
)  # 24 hours a day for an average number of days in 4 years
years_per_dataset = n_samples_h / (hours_per_year)
# Calculates frequencies per year
f_per_year = f_per_dataset / years_per_dataset

plt.step(f_per_year, np.abs(fft))
plt.xscale("log")
plt.ylim(0, 20000000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=["1/Year", "1/day"])
_ = plt.xlabel("Frequency (log scale)")

### Creating train, validation, test split for modeling purposes

# What do you need col indices for?
column_indices = {name: i for i, name in enumerate(aep_df.columns)}

# Train test split of data (TS data needs to be ordered so you split it in order)
n = len(aep_df)
train_df = aep_df[0 : int(n * 0.7)]
val_df = aep_df[int(n * 0.7) : int(n * 0.9)]
test_df = aep_df[int(n * 0.9) :]
num_features = aep_df.shape[1]

### Normalize data and visualize it

# Calculating mean and std
train_mean = train_df.mean()
train_std = train_df.std()
# Do this for each split - **Why do you use the training mean over the validation mean (look ahead bias?)
# Answer? To maintain consistency in normalization method across each split (Why is this better though?)
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Creates a graphic showing the distribution? of the different cols **Why do you apply this to the whole dataset
#   instead of just the training data set?
df_std = (aep_df - train_mean) / train_std
df_std = df_std.melt(var_name="Column", value_name="Normalized")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
## This plot provides a violin plot of the data (probability density after being normalized) which is a way to compare dist
#   among the different cols. We want to make sure there are no errors in the data like -9999 input (long tail)

# **What does the _ do?
# (_) is a convention in Python for a variable that is not going to be used, indicating that it is a temporary or throwaway variable.
_ = ax.set_xticklabels(aep_df.keys(), rotation=90)


#### Data Windowing function -- provided by Tensorflow -- Lots of setup
## Indexes and Offsets
# Create WindowGenerator Function
class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )


### This split window function from TF handles multiple columns for SO and MO
## Split into window of inputs and labels
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1,
        )

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window


### **This provides a plotting method which we use later?
## Plot method of split window
def plot(self, model=None, plot_col="mw", max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f"{plot_col} [normed]")
        plt.plot(
            self.input_indices,
            inputs[n, :, plot_col_index],
            label="Inputs",
            marker=".",
            zorder=-10,
        )

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(
            self.label_indices,
            labels[n, :, label_col_index],
            edgecolors="k",
            label="Labels",
            c="#2ca02c",
            s=64,
        )
        if model is not None:
            predictions = model(inputs)
            plt.scatter(
                self.label_indices,
                predictions[n, :, label_col_index],
                marker="X",
                edgecolors="k",
                label="Predictions",
                c="#ff7f0e",
                s=64,
            )

        if n == 0:
            plt.legend()

    plt.xlabel("Time [h]")


WindowGenerator.plot = plot


## **This creates a TF dataset needed for modeling?
# Method to convert to tf.data.Datasets
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,
    )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset

## **Last bit of setup for the window generator?
# Add properties to WindowGenerator Object


# Setting up for all the splits
@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


# **?
@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, "_example", None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

### Creating our own window function -- 24 hour windowing period


print("debug")
