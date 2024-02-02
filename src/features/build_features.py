import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from feature_definition import feature_build, date1


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df


def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def remove_object_columns(dataframe):
    # Remove columns with data type 'object'
    cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    dataframe = dataframe.drop(columns=cols)
    return dataframe

def final_check(dataframe):
    print("Data types after processing:")
    print(dataframe.dtypes)
    print("\nFirst few rows of the DataFrame:")
    print(dataframe.head())

if __name__ == "__main__":
    current_dir = Path(__file__)
    project_root = current_dir.parent.parent.parent  # Go up two levels to reach the project root
    train_path = project_root.as_posix() + "/data/raw/train.csv"
    test_path = project_root.as_posix() + "/data/raw/test.csv"
    output_path = project_root.as_posix() + "/data/processed"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    train_data.drop(columns="dropoff_datetime", inplace=True)
    train_data = feature_build(train_data, date1)
    
    test_data = feature_build(test_data, date1)
    # test_data = remove_object_columns(test_data)
    
    
    final_check(train_data)
    final_check(test_data)
    
    save_data(train_data, test_data, output_path)