from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

app = FastAPI()

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
    vendor_id: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float
    Month: float
    Pickup_Day: float = Field(..., alias='Pickup Day')
    Pickup_hours: float = Field(..., alias='Pickup hours')
    Pickup_mins: float = Field(..., alias='Pickup mins')
    distance: float

    class Config:
        allow_population_by_field_name = True


# Load the pre-trained RandomForest model
model_path = "models/trained_model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Extract features from input_data and make predictions using the loaded model
    features = [input_data.vendor_id,
                input_data.passenger_count,
                input_data.pickup_longitude,
                input_data.pickup_latitude,
                input_data.dropoff_longitude,
                input_data.dropoff_latitude,
                input_data.store_and_fwd_flag,
                input_data.Month,
                input_data.Pickup_Day,
                input_data.Pickup_hours,
                input_data.Pickup_mins,
                input_data.distance
                ]             
    prediction = model.predict([features])[0].item()
    # Return the prediction
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)