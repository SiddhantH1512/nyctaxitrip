from fastapi import FastAPI
from joblib import load
import pandas as pd
from datetime import datetime
from src.features.feature_definition import feature_build, date1
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    vendor_id: float
    pickup_datetime: str  # Using string to handle datetime
    dropoff_datetime: str
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float

# Load the pre-trained model and feature names
model_info = load("models/trained_model.joblib")
model = model_info["model"]
expected_feature_names = model_info["feature_names"]

@app.get("/")
def home():
    return {"message": "API is working fine"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert string datetime to actual datetime
        try:
            pickup_datetime = datetime.fromisoformat(input_data.pickup_datetime)
            dropoff_datetime = datetime.fromisoformat(input_data.dropoff_datetime)
        except ValueError as e:
            return {"error": f"Invalid datetime format: {str(e)}"}

        features = {
            "vendor_id": input_data.vendor_id,
            "pickup_datetime": pickup_datetime,
            "dropoff_datetime": dropoff_datetime,
            "passenger_count": input_data.passenger_count,
            "pickup_longitude": input_data.pickup_longitude,
            "pickup_latitude": input_data.pickup_latitude,
            "dropoff_longitude": input_data.dropoff_longitude,
            "dropoff_latitude": input_data.dropoff_latitude,
            "store_and_fwd_flag": input_data.store_and_fwd_flag
        }
        
        features_df = pd.DataFrame([features])
        features_df = feature_build(features_df, date1)

        # Reorder columns to match training data
        features_df = features_df[expected_feature_names]

        prediction = model.predict(features_df)[0]
        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
