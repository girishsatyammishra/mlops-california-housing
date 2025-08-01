from app.service import housingPricePredictionService
from fastapi import Request, Response
from fastapi import APIRouter, Query
from app.schema import HousePricePredictionInput
import pandas as pd

housing_price_prediction_router = APIRouter(
    prefix="",
    tags=["predict housing price"]
)

@housing_price_prediction_router.post("/predict")
def predict(input: HousePricePredictionInput):
    print(input)
    return housingPricePredictionService.predict(input)

@housing_price_prediction_router.get("/metrics")
def predict(request_body: dict):
    print(request_body)
    return {"name": "metrics"}