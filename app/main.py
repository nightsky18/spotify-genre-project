from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Spotify Genre Classifier")

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR.parent / "models" / "gb_pipeline.pkl")

class SongInput(BaseModel):
    track_popularity: float
    danceability: float
    energy: float
    key: float
    loudness: float
    mode: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: float

@app.get("/")
def home():
    return {"message": "API activa"}

@app.post("/predict")
def predict(data: SongInput):
    df_input = pd.DataFrame([data.model_dump()])
    pred = model.predict(df_input)[0]
    return {"playlist_genre": pred}