import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
# from scipy.stats import boxcox, yeo_johnson, quantile

from configs.config import PROCESSED_DATA_DIR

# TRANSFORM_DICT = {
#     "log": lambda x: np.log(x),
#     "sqrt": lambda x: np.sqrt(x),
#     "boxcox": lambda x: boxcox(x),
#     "yeo-johnson": lambda x: yeo_johnson(x),
#     "quantile": lambda x: quantile(x),
# }
def load_dataset(dataset_name, test_size=0.2, transform=False):
    data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, dataset_name))
    data.set_index('Date', inplace=True)
    
    y = data['target']
    X = data.drop(columns=[
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dollar Volume',
        'return_1d', 'log_return_1d', 'target'
    ], errors='ignore')


    # Sécurité pour les NaN/inf
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.dropna(inplace=True)
        y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    if transform:
        preprocessing = transform_pipeline()
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.transform(X_test)

    # Ensure clean, numeric, contiguous arrays for downstream learners
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    y_test = np.asarray(y_test, dtype=np.float32).ravel()

    # Replace any remaining non-finite values defensively
    # X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    # X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # mulvar_train, target_train = X[:"2014-12"].to_numpy(), y[:"2014-12"].to_numpy()
    # mulvar_valid, target_valid = X["2015-01":"2016-01"].to_numpy(), y["2015-01":"2016-01"].to_numpy()
    # mulvar_test, target_test = X["2016-02":].to_numpy(), y["2016-02":].to_numpy()


    # scaler = StandardScaler()
    # mulvar_train = scaler.fit_transform(mulvar_train)
    # mulvar_valid = scaler.transform(mulvar_valid)
    # mulvar_test = scaler.transform(mulvar_test)

    # Convert targets from -1/1 to 0/1 for BCEWithLogitsLoss
    y_train = (y_train + 1) / 2
    y_test = (y_test + 1) / 2
    return (X_train, y_train), (X_test, y_test)

    lookback =  54
    lookahead = 1

    train_dataset = TimeSeriesDataset(mulvar_train, lookback, lookahead, target=target_train_bce)
    valid_dataset = TimeSeriesDataset(mulvar_valid, lookback, lookahead, target=target_valid_bce)
    test_dataset = TimeSeriesDataset(mulvar_test, lookback, lookahead, target=target_test_bce)

    train_dataloader = train_dataset.to_dataloader(shuffle=True)
    valid_dataloader = valid_dataset.to_dataloader(shuffle=False)
    test_dataloader = test_dataset.to_dataloader(shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

class TimeSeriesDataset:
    def __init__(self, data, lookback, lookahead, target=None, stride=1, batch_size=32):
        self.data = data
        self.lookback = lookback
        self.lookahead = lookahead
        self.stride = stride
        self.batch_size = batch_size
        self.target = target

    def to_dataloader(self, shuffle=False):
        """
        Convert the data to a PyTorch times series dataset and return a DataLoader.
        """
        input, output = self.to_timeseries_input(self.data)
        dataset = TensorDataset(input, output)
        data_loader = DataLoader(dataset, 
                                                    batch_size=self.batch_size, 
                                                    shuffle=shuffle)
        return data_loader

    def rolling_window(self, series, window_size):
        """
        Given a sequence, return a numpy array of the rolling window.
        """
        arr = []
        for i in range(0, series.shape[0] - window_size + 1, self.stride):
            arr.append(series[i : (i + window_size)])
        return np.array(arr)

    def to_timeseries_input(self, series):
        """
        Given a sequence, return a PyTorch tensor couple (inputs, outputs) 
        of the rolling window.
        """
        inputs = self.rolling_window(series[:-self.lookahead], self.lookback)
        if self.target is not None:
            outputs = self.rolling_window(self.target[self.lookback:], self.lookahead)
        else:
            outputs = self.rolling_window(series[self.lookback:], self.lookahead)

        return torch.tensor(inputs, dtype=torch.float32), \
                 torch.tensor(outputs, dtype=torch.float32).unsqueeze(-1)

        
   


def transform_pipeline():
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler())
    preprocessing = ColumnTransformer([
            ("log", log_pipeline, ["GKV", "VIX_Close"]),
        ],
    remainder=default_num_pipeline)  
    return preprocessing