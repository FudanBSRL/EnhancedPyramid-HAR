# ==============================================================================
# Fdu Bioinspired Structure and Robot Lab

# Usage:
# Run this script with command line arguments specifying the file path for
# the dataset, number of time steps in a segment, and the step size for segmenting.
# Example:
# python this_script.py --path "\WISDM_ar_v1.1_raw.txt" --time_steps 200 --step 50
# ==============================================================================

# Importing necessary libraries
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import numpy as np
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Function to read and preprocess data
def read_data(filepath):
    # Reading data with specific column names
    df = read_csv(filepath, header=None, names=['user-id', 'activity', 'timestamp', 'X', 'Y', 'Z'], error_bad_lines=False)
    # Cleaning the 'Z' column and converting it to float
    df['Z'] = df['Z'].str.replace(';', '').astype(float)
    return df

# Function to segment the data
def segment_window(df, time_steps, step):
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['X'].values[i:i + time_steps]
        ys = df['Y'].values[i:i + time_steps]
        zs = df['Z'].values[i:i + time_steps]
        segments.append([xs, ys, zs])
        label = mode(df['activityEncode'].values[i:i + time_steps])[0][0]
        labels.append(label)
    return np.asarray(segments, dtype=np.float32), np.asarray(labels)

def main():
    print("Script started. Processing data...")

    # Setting the CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, default=200, help='Number of time steps in a segment')
    parser.add_argument('--step', type=int, default=50, help='Step size for segmenting the dataset')
    parser.add_argument('--path', type=str, default="\WISDM_ar_v1.1_raw.txt", help='File path for the dataset')
    args = parser.parse_args()

    # Reading the dataset
    df = read_data(args.path)

    # Encoding activity labels
    label_encoder = LabelEncoder()
    df['activityEncode'] = label_encoder.fit_transform(df['activity'])

    # Dropping NaN values
    df.dropna(axis=0, inplace=True)

    # Normalizing X, Y, Z columns
    for axis in ['X', 'Y', 'Z']:
        df[axis] = (df[axis] - df[axis].min()) / (df[axis].max() - df[axis].min())

    # Segmenting the dataset
    segments, labels = segment_window(df, args.time_steps, args.step)

    # Splitting the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, shuffle=True)
    print("Now x_train.shape: {}, y_train.shape : {}".format(x_train.shape, x_test.shape))

    # Saving the segmented data
    np.save('WISDM_x_train.npy', x_train)
    np.save('WISDM_x_test.npy', x_test)
    np.save('WISDM_y_train.npy', y_train)
    np.save('WISDM_y_test.npy', y_test)

    print("Data processing complete. Files saved.")

if __name__ == "__main__":
    main()
