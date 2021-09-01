import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw
from scipy.stats import anderson
from torch.utils.data import Dataset
from torch import tensor


def prepare_df(args):
    """
    args[0] = path, args[1] = column name, args[2] = replace_commas flag, args[3] = reverse flag
    load file from csv with given path
    process data from given column
    replace_commas flag for parsing data like '123,345'
    reverse flag for parsing data if dataset is in descending order
    :return data: df with a processed data as a single column
    """
    path, column, replace_commas, reverse = args
    data = pd.read_csv(path)
    data = data[[column]]
    if reverse:
        data.index = data.index[::-1]
        data = data.iloc[::-1]
    if replace_commas:
        data[column] = data[column].apply(lambda x: float(x.replace(',', '')))
    else:
        data[column] = data[column].apply(lambda x: float(x))
    return data


def add_log_return(data):
    """
    compute log_return for data in given column name
    :param data: input df
    :return: df with new log_ret column
    """
    data['log_ret'] = np.log(data[data.columns[0]]) - np.log(data[data.columns[0]].shift(1))
    data = data[1:]
    return data


def scale_data(data):
    """
    apply standard scaler to log_ret column
    :param data: input df
    :return: df with new log_scaled column
    """
    scaler = StandardScaler()
    log_scaled = scaler.fit_transform(data['log_ret'].values.reshape(-1, 1))
    data['log_scaled'] = log_scaled
    return data


def apply_inverse_lambertw(data):
    """
    apply inverse lambert w transformation and scale resulting data
    inverse lambert w algorithm taken from: https://arxiv.org/pdf/1010.2265.pdf
    coefficient delta is estimated by taking best results according to anderson test (not sure if it is right thing to do)
    main idea is to have most normal-looking distribution as a result of transformation
    :param data: input df
    :return: data with new t_scaled column (applied normalization x inverse lambert w transform)
    """
    scaler = StandardScaler()
    stat = math.inf
    delta_opt = 0
    for delta in range(1, 20000):
        t = np.sign(data['log_ret']) * np.sqrt((lambertw(data['log_ret'] * data['log_ret'] * delta) / delta)).astype(
            float)
        if anderson(t, dist='norm')[0] < stat:
            stat = anderson(t, dist='norm')[0]
            delta_opt = delta
        else:
            break
    t = np.sign(data['log_ret']) * np.sqrt(
        (lambertw(data['log_ret'] * data['log_ret'] * delta_opt) / delta_opt)).astype(float)
    t_scaled = scaler.fit_transform(t.values.reshape(-1, 1))
    data['t_scaled'] = t_scaled
    return data


def add_rolling(data):
    """
    apply a rolling window of length 127, take mean
    :param data:input df
    :return: data with new collumn t_rolling
    """
    t_rolling = data['t_scaled'].rolling(127).mean()
    data['t_rolling'] = t_rolling
    return data


def chain_f(start, funcs):
    """
    utility func to run whole preprocessing
    """
    res = start
    for func in funcs:
        res = func(res)
    return res

# functions to apply during preprocessing
PIPE = [prepare_df, add_log_return, scale_data, apply_inverse_lambertw, add_rolling]


def prepare_dataset(path, column, replace_commas=True, reverse=True):
    """
    :param path: path to csv file with financial timeseries data
    :param column: process data from given column
    :param replace_commas: flag for parsing data like '123,345'
    :param reverse:flag to use if dataset is in a descending order
    :return data: df with a processed data as a single column
    """
    return chain_f((path, column, replace_commas, reverse), PIPE)


class TimeseriesLoader(Dataset):
    """
    Simple dataset which chunks initial financial time-series
    into smaller sub-time-series of constant length
    """

    def __init__(self, ts, length=127):
        assert len(ts) >= length, 'provide shorter length for given timeseries'
        self.ts = ts
        self.length = length

    def __len__(self):
        return max(len(self.ts) - self.length, 0)

    def __getitem__(self, idx):
        return tensor(self.ts[idx:idx + self.length].values).reshape(-1, self.length)

