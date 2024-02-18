# ==============================================================================
# Fdu Bioinspired Structure and Robot Lab
# Usage:
# Run this script with command line arguments specifying the window size, step size,
# and the base path for the dataset.
# Example:
# python this_script.py --window_size 128 --step_size 64 --data_path "./PAMAP2_Dataset/Protocol"
# ==============================================================================

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def read_and_preprocess(subject_id, columns, base_path):
    """
    Read and preprocess the dataset for a given subject.

    Args:
    subject_id (int): ID of the subject.
    columns (list): List of columns to be selected from the dataset.
    base_path (str): Base path to the dataset files.

    Returns:
    Tuple of numpy.ndarray: Returns the sensor data and corresponding labels.
    """
    filepath = f"{base_path}/subject10{subject_id}.dat"
    data = pd.read_csv(filepath, header=None, sep=' ', usecols=columns).dropna().values
    return data[:, 1:], data[:, 0].astype(int)


def window(data, labels, size, stride):
    """
    Create windows of data with the corresponding label.

    Args:
    data (numpy.ndarray): The dataset.
    labels (numpy.ndarray): The labels corresponding to the dataset.
    size (int): The window size.
    stride (int): The stride between consecutive windows.

    Returns:
    Tuple of lists: Returns the segmented data and their labels.
    """
    x, y = [], []
    for i in range(0, len(labels), stride):
        if i + size >= len(labels):
            break
        window_labels = set(labels[i:i + size])
        if len(window_labels) == 1 and labels[i] != 0:
            x.append(data[i: i + size])
            y.append(labels[i])
    return x, y


def generate_samples(window_size, step, columns, base_path):
    """
    Generate samples and labels for the given window size and step.

    Args:
    window_size (int): Size of the data window.
    step (int): Step size for the window.
    columns (list): List of columns to be used from the dataset.
    base_path (str): Base path to the dataset files.

    Returns:
    Tuple of numpy.ndarray: Returns all data windows and their labels.
    """
    X, Y = [], []
    for subject_id in range(1, 10):
        data, labels = read_and_preprocess(subject_id, columns, base_path)
        x, y = window(data, labels, window_size, step)
        X.extend(x)
        Y.extend(y)

    category_indices = sorted(set(Y))
    Y = [category_indices.index(label) for label in Y]
    return X, np.array(Y)


def main():
    parser = argparse.ArgumentParser(description='HAR Data Processing')
    parser.add_argument('--window_size', type=int, default=128, help='Window size for data segmentation')
    parser.add_argument('--step_size', type=int, default=64, help='Step size for data segmentation')
    parser.add_argument('--data_path', type=str, default='./PAMAP2_Dataset/Protocol', help='Base path for the dataset')
    args = parser.parse_args()

    columns = [1] + list(range(4, 16)) + list(range(21, 33)) + list(range(38, 50))
    X, Y = generate_samples(args.window_size, args.step_size, columns, args.data_path)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    window_size_str = str(args.window_size)
    np.save(f'./x_train_windows_{window_size_str}', x_train)
    np.save(f'./x_test_windows_{window_size_str}', x_test)
    np.save(f'./y_train_windows_{window_size_str}', y_train)
    np.save(f'./y_test_windows_{window_size_str}', y_test)
    print(f"Data processing and saving complete. Window size: {window_size_str}")


if __name__ == '__main__':
    main()
