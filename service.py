from fastapi import FastAPI
from joblib import load
import numpy as np
import pandas as pd
from datetime import datetime
from src.features.feature_definition import feature_build, date1
from pydantic import BaseModel, Field

app = FastAPI()


class PredictionInput(BaseModel):
    vendor_id: float
    pickup_datetime: datetime
    # dropoff_datetime: datetime
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float
    
    class Config:
        populate_by_name = True

model = load("models/trained_model.joblib")
# expected_feature_names = model["feature_names"]

@app.get("/")
def home():
    return {"message": "API is working fine"}
            
 
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        
        features = [
            float(input_data.vendor_id),
            input_data.pickup_datetime,  
            float(input_data.passenger_count),
            float(input_data.pickup_longitude),
            float(input_data.pickup_latitude),
            float(input_data.dropoff_longitude),
            float(input_data.dropoff_latitude),
            float(input_data.store_and_fwd_flag)
            
        ]
        
        features = pd.DataFrame([features], columns=[
            'vendor_id', 'pickup_datetime', 'passenger_count',
            'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'store_and_fwd_flag'
        ])
        features = feature_build(features, date1)
        
        print(f"Number of features after feature_build: {features.shape[1]}")
        if features.shape[1] != 9:
             print("Feature names:", features.columns)
        
        features_array = features.to_numpy()
        
        prediction = model.predict(features_array)[0]
        # Convert numpy data types to Python native type for JSON serialization
        if isinstance(prediction, np.number):
            prediction = prediction.item()

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)



