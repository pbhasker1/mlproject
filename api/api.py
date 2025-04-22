from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the pre-trained model
model = joblib.load("pretrained_model.pkl")

@app.get("/")
def read_root():
    return {"message": "API is working"}

@app.post("/predict/")
def predict(features: dict):
    # Parse input features and make predictions
    input_data = [features["data"]]  # Ensure data is in the correct format
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}