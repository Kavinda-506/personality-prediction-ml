from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

model = joblib.load("model/model.pkl")

# Label mapping
LABEL_MAP = {0: "Extrovert", 1: "Introvert"}


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(data: dict):

    input_data = np.array([
        data["Time_spent_Alone"],
        data["Stage_fear"],
        data["Social_event_attendance"],
        data["Going_outside"],
        data["Drained_after_socializing"],
        data["Frineds_circle_size"],
        data["Post_frequency"]
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    result = LABEL_MAP.get(int(prediction[0]), "Unknown")

    return {
        "prediction": result
    }