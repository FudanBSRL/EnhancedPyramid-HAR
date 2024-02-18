import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import os
import zipfile
import argparse
import numpy as np
import _pickle as cp
from io import BytesIO
from pandas import Series
import os
import sys
import warnings
import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import torch
from torch import nn
import torch.nn.functional as F
from thop import profile
# Constants for the OPPORTUNITY challenge
# Suppress warnings
warnings.filterwarnings('ignore')

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Constants for sensor data
NB_SENSOR_CHANNELS = 113
SLIDING_WINDOW_LENGTH = 64
SLIDING_WINDOW_STEP = 8
NUM_CLASSES = 18

# Add 'src' directory to import path for modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)

OPPORTUNITY_DATA_FILES = ['OpportunityUCIDataset/dataset/S1-Drill.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S2-Drill.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S3-Drill.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL5.dat'
                          ]
# Hardcoded thresholds for normalization
NORM_MAX_THRESHOLDS = [3000] * 39 + [10000] * 9 + [1500] * 6 + [5000] * 6 + [10000] * 6 + [250, 25, 200, 5000] * 2 + [10000] * 6 + [250] * 2
NORM_MIN_THRESHOLDS = [-x for x in NORM_MAX_THRESHOLDS]

def select_columns_opp(data):
    """Select the required columns from OPPORTUNITY challenge dataset."""
    features_delete = np.concatenate([np.arange(start, start + 4) for start in range(46, 99, 13)])
    features_delete = np.concatenate([features_delete, np.arange(134, 249)])
    return np.delete(data, features_delete, 1)

def normalize(data):
    """Normalize sensor data."""
    diffs = np.array(NORM_MAX_THRESHOLDS) - np.array(NORM_MIN_THRESHOLDS)
    for i, (max_val, min_val) in enumerate(zip(NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)):
        data[:, i] = (data[:, i] - min_val) / diffs[i]
    np.clip(data, 0, 0.99, out=data)
    return data

def divide_x_y(data, label):
    """Divide dataset into features and labels."""
    data_x = data[:, 1:114]
    label_map = {
        'locomotion': 114,
        'gestures': 115
    }
    if label not in label_map:
        raise RuntimeError("Invalid label: '%s'" % label)
    data_y = data[:, label_map[label]]
    return data_x, data_y


def check_data(data_set):
    """Try to access to the file and checks if dataset is in the data directory
       In case the file is not found try to download it from original location
    :param data_set:
            Path with original OPPORTUNITY zip file
    :return:
    """
    print('Checking dataset {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print('... dataset path {0} not found'.format(data_set))
        import urllib
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print('... creating directory {0}'.format(data_dir))
            os.makedirs(data_dir)
        print('... downloading data from {0}'.format(origin))
        urllib.request.urlretrieve(origin, data_set)

    return data_dir

def process_dataset_file(data, label):
    """Process individual OPPORTUNITY files."""
    data = select_columns_opp(data)
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    data_x[np.isnan(data_x)] = 0
    return normalize(data_x), data_y.astype(int)

def adjust_idx_labels(data_y, label):
    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y

def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters:
        shape (int or tuple): Shape to be normalized.

    Returns:
        tuple: Normalized shape tuple.
    """
    if isinstance(shape, int):
        return (shape,)
    elif isinstance(shape, tuple):
        return shape
    else:
        raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(array, window_size, step_size=None, flatten=True):
    """
    Return a sliding window over an array in any number of dimensions.

    Parameters:
        array (numpy.ndarray): An n-dimensional numpy array.
        window_size (int or tuple): Size of each dimension of the window.
        step_size (int or tuple, optional): Step size for each dimension. Defaults to window_size.
        flatten (bool): If True, all slices are flattened.

    Returns:
        numpy.ndarray: Array containing each n-dimensional window from the input array.
    """
    if step_size is None:
        step_size = window_size

    window_size, step_size = np.array(norm_shape(window_size)), np.array(norm_shape(step_size))
    shape = np.array(array.shape)

    if not (len(shape) == len(window_size) == len(step_size)):
        raise ValueError('array.shape, window_size and step_size must all have the same length.')

    if np.any(window_size > shape):
        raise ValueError('window_size cannot be larger than array in any dimension.')

    newshape = ((shape - window_size) // step_size) + 1
    newshape = np.concatenate([newshape, window_size])

    newstrides = np.array(array.strides) * step_size
    newstrides = np.concatenate([newstrides, array.strides])

    strided_array = ast(array, shape=newshape, strides=newstrides)

    if flatten:
        return strided_array.reshape(-1, *window_size)
    return strided_array

def generate_data():

    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels
    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    # data_dir = check_data(dataset)

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))

    zf = zipfile.ZipFile(".\OpportunityUCIDataset.zip")
    print('Processing dataset files ...')
    for filename in OPPORTUNITY_DATA_FILES:
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {0}'.format(filename))
            x, y = process_dataset_file(data, "gestures")
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(filename))

    # Dataset is segmented into train and test
    nb_training_samples = 557963
    # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    X_train, y_train = data_x[:nb_training_samples,:], data_y[:nb_training_samples]
    X_test, y_test = data_x[nb_training_samples:,:], data_y[nb_training_samples:]
    print(X_train.shape ,y_train.shape)
    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    obj = [(X_train, y_train), (X_test, y_test)]
    f = open(os.path.join(".\opportunity", "oppChallenge_gestures.data"), 'wb')
    cp.dump(obj, f, protocol=-1)
    f.close()

def get_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess OPPORTUNITY dataset')
    parser.add_argument('-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument('-o', '--output', type=str, help='Processed data file', required=True)
    parser.add_argument('-t', '--task', type=str.lower, help='Type of activities to be recognized', default="gestures", choices=["gestures", "locomotion"], required=False)
    return parser.parse_args()
def load_dataset(filename):
    """
    Load dataset from a file.
    """
    with open(filename, 'rb') as f:
        data = cp.load(f)

    X_train, y_train = data[0]
    X_test, y_test = data[1]
    print(f" ..from file {filename}")
    print(f" ..reading instances: train {X_train.shape}, test {X_test.shape}")

    return X_train.astype(np.float32), y_train.astype(np.uint8), X_test.astype(np.float32), y_test.astype(np.uint8)

def opp_sliding_window(data_x, data_y, window_size, step_size):
    """
    Apply sliding window mechanism to the dataset.
    """
    data_x = sliding_window(data_x, (window_size, data_x.shape[1]), (step_size, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, window_size, step_size)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def convert_to_one_hot(y, num_classes):
    """
    Convert labels to one-hot encoding.
    """
    return np.eye(num_classes)[y.reshape(-1)]

# Loading and processing the data


if __name__ == '__main__':
    args = get_args()
    generate_data(args.input, args.output, args.task)
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('./oppChallenge_gestures.data')

    assert NB_SENSOR_CHANNELS == X_train.shape[1]

    # Applying sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    # Reshaping data for Conv1D and one-hot encoding
    X_train = X_train.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    X_test = X_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    y_train = convert_to_one_hot(y_train, NUM_CLASSES)
    y_test = convert_to_one_hot(y_test, NUM_CLASSES)

    print(f" ..after sliding and reshaping, train data: inputs {X_train.shape}, targets {y_train.shape}")
    print(f" ..after sliding and reshaping, test data: inputs {X_test.shape}, targets {y_test.shape}")

    # Saving the processed data
    np.save(r".\train_x.npy", X_train)
    np.save(r".\train_y.npy", y_train)
    np.save(r".\test_x.npy", X_test)
    np.save(r".\test_y.npy", y_test)
