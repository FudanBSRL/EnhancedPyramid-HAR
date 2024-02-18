# ==============================================================================
# Fdu Bioinspired Structure and Robot Lab
#

# Usage:
# Run this script with command line arguments specifying the file paths for
# training and testing data directories, and y_train and y_test files.
# Example:
# python this_script.py --train_path "path/to/train_data" --test_path "path/to/test_data" ...
# =============================================================================

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

def make_x(path):
    """
    Reads and concatenates files in the given directory, transposing the result.

    Parameters:
    path (str): Path to the directory containing the files.

    Returns:
    numpy.ndarray: Transposed array of the concatenated files.
    """
    all_file_names = os.listdir(path)
    x = [np.genfromtxt(os.path.join(path, file_name)) for file_name in all_file_names]
    return np.transpose(x, (1, 2, 0))

def main():
    print("Starting data processing...")

    # Parsing command line arguments for file paths
    parser = argparse.ArgumentParser(description='Process UCI HAR Dataset')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--y_train_path', type=str, required=True, help='Path to the y_train.txt file')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the y_test.txt file')
    args = parser.parse_args()

    # Processing training and testing data
    train_x = make_x(args.train_path)
    test_x = make_x(args.test_path)
    train_y = np.genfromtxt(args.y_train_path, dtype=int).reshape(-1)
    test_y = np.genfromtxt(args.y_test_path, dtype=int).reshape(-1)

    # Saving the processed data
    np.save('./x_train.npy', train_x)
    np.save('./x_test.npy', test_x)
    np.save('./y_train.npy', train_y)
    np.save('./y_test.npy', test_y)

    print("Data processing complete. Files saved.")

if __name__ == '__main__':
    main()