import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import pandas as pd

date1 = "pickup_datetime"


def build_features(dataframe, date1):
    dataframe[date1] = pd.to_datetime(dataframe[date1])
    dataframe["Month"] = dataframe[date1].dt.month
    dataframe["Pickup Day"] = dataframe[date1].dt.dayofweek
    dataframe["Pickup hours"] = dataframe[date1].dt.hour
    dataframe["Pickup mins"] = dataframe[date1].dt.minute

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

def test_created_features(dataframe, date1):
    build_features(dataframe, date1)
    create_distance_feature(dataframe)
    print(dataframe.head())
    
def feature_build(dataframe, date1):
    build_features(dataframe, date1)
    create_distance_feature(dataframe)
    do_not_use_for_training = ["id", "pickup_datetime", "dropoff_datetime"]
    feature_names = [feature for feature in dataframe.columns if feature not in do_not_use_for_training]
    print(f"We have {len(feature_names)} features in {dataframe}")
    return dataframe[feature_names]

if __name__ == "__main__":
    current_dir = Path(__file__)
    project_root = current_dir.parent.parent.parent  # Go up two levels to reach the project root
    datapath = project_root.as_posix() + "/data/raw/test.csv"
    data = pd.read_csv(datapath)
    test_created_features(data, date1)

