import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd


date1 = "pickup_datetime"


def build_features(dataframe, date1):
    dataframe[date1] = pd.to_datetime(dataframe[date1])
    dataframe["Month"] = dataframe[date1].dt.month
    dataframe["Pickup Day"] = dataframe[date1].dt.dayofweek
    dataframe["Pickup hours"] = dataframe[date1].dt.hour
    dataframe["Pickup mins"] = dataframe[date1].dt.minute
    dataframe["Day Name"] = dataframe[date1].dt.day_name()
    
    # Handling dropoff_datetime
    if 'dropoff_datetime' in dataframe.columns:
        # dataframe["dropoff_datetime"] = pd.to_datetime(dataframe["dropoff_datetime"])
        # dataframe["Dropoff Month"] = dataframe["dropoff_datetime"].dt.month
        # dataframe["Dropoff Day"] = dataframe["dropoff_datetime"].dt.dayofweek
        # dataframe["Dropoff hours"] = dataframe["dropoff_datetime"].dt.hour
        # dataframe["Dropoff mins"] = dataframe["dropoff_datetime"].dt.minute
        # dataframe["Day Name"] = dataframe[date1].dt.day_name()
        dataframe.drop(['dropoff_datetime'], axis=1, inplace=True)
        
    if 'store_and_fwd_flag' in dataframe.columns:
        dataframe['store_and_fwd_flag'] = dataframe['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

    return dataframe


def distances(row):
    r = 6371
    
    lon1 = radians(row["pickup_longitude"])
    lat1 = radians(row["pickup_latitude"])
    lon2 = radians(row["dropoff_longitude"])
    lat2 = radians(row["dropoff_latitude"])
    
    diff_lon = lon2 - lon1
    diff_lat = lat2 - lat1

    a = sin(diff_lat/2) ** 2 + cos(lat1) * cos(lat2) * sin(diff_lon/2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Return the calculated distance
    return r * c

def create_distance_feature(dataframe):
    dataframe["distance"] = dataframe.apply(distances, axis=1)
    return dataframe
    
def scale(dataframe):
    # Drop unnecessary columns
    columns_to_drop = ["id", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    dataframe.drop(columns=[col for col in columns_to_drop if col in dataframe.columns], inplace=True)
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    scaler.set_output(transform="pandas")
    dataframe = scaler.fit_transform(dataframe)
    return dataframe

def test_created_features(dataframe, date1):
    build_features(dataframe, date1)
    create_distance_feature(dataframe)

    # Print DataFrame after creating distance feature
    print("DataFrame after creating distance feature:\n", dataframe.head())

    # Directly apply encoding for 'Day Name'
    day_name_dummies = pd.get_dummies(dataframe['Day Name'], prefix='Day', dtype=int)
    dataframe = pd.concat([dataframe, day_name_dummies], axis=1)
    dataframe.drop(['Day Name'], axis=1, inplace=True)

    # Print DataFrame after encoding 'Day Name'
    print("DataFrame after encoding 'Day Name':\n", dataframe.head())

    scale(dataframe)
    print("DataFrame after scaling:\n", dataframe.head())

    
def feature_build(dataframe, date1):
    dataframe = build_features(dataframe, date1)
    dataframe = create_distance_feature(dataframe)
    day_name_dummies = pd.get_dummies(dataframe['Day Name'], prefix='Day', dtype=int)
    dataframe = pd.concat([dataframe, day_name_dummies], axis=1)
    dataframe.drop(['Day Name'], axis=1, inplace=True)
    dataframe = scale(dataframe)
    return dataframe

    
    

if __name__ == "__main__":
    current_dir = Path(__file__)
    project_root = current_dir.parent.parent.parent  # Go up two levels to reach the project root
    datapath = project_root.as_posix() + "/data/raw/test.csv"
    data = pd.read_csv(datapath)
    test_created_features(data, date1)