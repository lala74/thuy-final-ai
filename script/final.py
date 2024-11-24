import math
import os
import sys
from functools import total_ordering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

feature_columns = ["Low", "Open", "High", "Close"]
number_features = len(feature_columns)


def is_csv_file(file_path) -> bool:
    """Check if the given file is a CSV file."""
    return file_path.endswith(".csv")


def is_data_has_more_values_than_threshold(data, threshold=120):
    return data.size > threshold


def is_missing_data_greater_than_threshold(data, feature_columns, threshold=0.1):
    # Return total missing values and total number of values
    total_missing = 0
    total_values = data.size
    for feature in feature_columns:
        missing_indices = data[data[feature].isnull()].index
        num_missing = len(missing_indices)

        if num_missing > 0:
            total_missing += num_missing

    missing_percentage = total_missing / total_values
    return missing_percentage > threshold


def print_valid_files(folder_path):
    """Walk through all files in the given folder."""
    valid_files = []
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    for root, _, files in os.walk(folder_path):
        files = sorted(files)
        for file in files:
            file_path = os.path.join(root, file)

            if not is_csv_file(file_path):
                continue

            data = pd.read_csv(file_path)
            if not is_data_has_more_values_than_threshold(data, 120):
                print(f"Data in {file_path} is too small.")
                continue

            if is_missing_data_greater_than_threshold(data, feature_columns, 0.1):
                print(f"Missing data in {file_path}")
                continue

            valid_files.append(file_path)

    # Print valid files, each file in a new line
    print("Valid files:")
    print("\n".join(valid_files))


def fill_missing_values(data):
    """
    Fill missing values in the given dataframe.

    Args:
        data (pd.DataFrame): The dataframe to fill missing values.
    """
    # Fill NaN values
    # 1. Interpolate linearly for middle NaNs
    data = data.interpolate(method="linear", axis=0)
    # 2. Fill the first NaNs with the next non-NaN value
    data = data.fillna(method="bfill", axis=0)
    # 3. Fill the last NaNs with the last non-NaN value
    data = data.fillna(method="ffill", axis=0)

    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    print_valid_files(folder_path)
    data = pd.read_csv("data.csv")
    data = fill_missing_values(data)
