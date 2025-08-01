from fastapi import FastAPI, Depends, HTTPException
from config import Config
from app.controller import housing_price_prediction_controller
import mlflow

def create_app():
    app = FastAPI()
    # CORS(app)
    register_routes(app, Config.BASE_URL, Config.API_VERSION)

    return app


def register_routes(app, base_url, api_version):
    app.include_router(housing_price_prediction_controller.housing_price_prediction_router, prefix=f"/{base_url}/{api_version}/housing-price")
