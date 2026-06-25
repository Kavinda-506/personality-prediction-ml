# Personality Prediction ML

A machine learning API that predicts whether a person is an **Introvert** or **Extrovert** based on behavioral features. Built with a Random Forest Classifier and served via FastAPI.

## Project Structure

```
├── app.py                  # FastAPI application
├── data/
│   ├── personality_dataset.csv   # Raw dataset
│   └── cleaned_dataset.csv       # Preprocessed dataset
├── model/
│   ├── model.pkl                 # Trained Random Forest model
│   └── label_encoder.pkl         # Label encoder
├── notebook/
│   ├── preprocessing.ipynb       # Data cleaning & preprocessing
│   └── training.ipynb            # Model training & evaluation
├── requirements.txt
├── Procfile                      # Deployment config
└── .gitignore
```

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `Time_spent_Alone` | Float (0–10) | Hours spent alone daily |
| `Stage_fear` | Binary (0/1) | Has stage fear |
| `Social_event_attendance` | Float (0–10) | Frequency of attending social events |
| `Going_outside` | Float (0–10) | Frequency of going outside |
| `Drained_after_socializing` | Binary (0/1) | Feels drained after socializing |
| `Frineds_circle_size` | Float (0–15) | Size of friend circle |
| `Post_frequency` | Float (0–10) | Social media posting frequency |

## Setup

```bash
# Clone the repository
git clone https://github.com/Kavinda-506/personality-prediction-ml.git
cd personality-prediction-ml

# Create virtual environment
python -m venv venv
venv\Scripts\activate          

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### Health Check

```
GET /
```

**Response:**
```json
{ "status": "ok" }
```

### Predict Personality

```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Time_spent_Alone": 9.0,
  "Stage_fear": 1,
  "Social_event_attendance": 0.0,
  "Going_outside": 0.0,
  "Drained_after_socializing": 1,
  "Frineds_circle_size": 0.0,
  "Post_frequency": 3.0
}
```

**Response:**
```json
{ "prediction": "Introvert" }
```

### Example using cURL

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time_spent_Alone":2.0,"Stage_fear":0,"Social_event_attendance":6.0,"Going_outside":7.0,"Drained_after_socializing":0,"Frineds_circle_size":12.0,"Post_frequency":7.0}'
```

### Example using Python

```python
import requests

response = requests.post("http://127.0.0.1:8000/predict", json={
    "Time_spent_Alone": 2.0,
    "Stage_fear": 0,
    "Social_event_attendance": 6.0,
    "Going_outside": 7.0,
    "Drained_after_socializing": 0,
    "Frineds_circle_size": 12.0,
    "Post_frequency": 7.0
})

print(response.json())  # {"prediction": "Extrovert"}
```

## Tech Stack

- **Python**
- **FastAPI** — API framework
- **scikit-learn** — Random Forest Classifier
- **NumPy / Pandas** — Data processing
- **Joblib** — Model serialization
- **Uvicorn** — ASGI server
