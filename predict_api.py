from fastapi import FastAPI, UploadFile, File
import pandas as pd
import mlflow.pyfunc
import os
import uuid

app = FastAPI()

# Use your actual Run ID here
RUN_ID = "2d06aa6bbe294bffaa1c564e8d5a89d2"
MODEL_URI = f"runs:/{RUN_ID}/model"

model = mlflow.pyfunc.load_model(MODEL_URI)

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/prediction_log.csv"

@app.get("/")
def home():
    return {"message": "California Housing Predictor is live!"}

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    predictions = model.predict(df)
    df["prediction"] = predictions
    df["uuid"] = str(uuid.uuid4())
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
    return {"predictions": predictions.tolist()}
