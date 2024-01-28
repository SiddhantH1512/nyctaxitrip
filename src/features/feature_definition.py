import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import pandas as pd

def build_features(dataframe, date1):
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

if __name__ == "__main__":
    # Use pathlib.Path to handle paths in a platform-independent way
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent  # Go up two levels to reach the project root
    datapath = project_root / "data" / "raw" / "test.csv"

    date1 = "pickup_datetime"
    # Read the data
    data = pd.read_csv(datapath)

    # Assuming 'date1' is defined somewhere in your script
    test_created_features(data, date1)
