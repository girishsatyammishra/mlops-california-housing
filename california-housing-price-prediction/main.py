from fastapi.responses import RedirectResponse
from config import Config
from datetime import datetime
import uvicorn
from app import create_app
import mlflow

# Set MLflow experiment on app start
experiment_name = Config.MLFLOW_EXPERIMENT
mlflow.set_experiment(experiment_name)

app = create_app()


@app.on_event("startup")
async def startup_event():
    print(f' ***** App Running')


@app.get('/')
def redirect_to():
    redirect_url = f'/{Config.BASE_URL}/{Config.API_VERSION}/info'
    return RedirectResponse(url=redirect_url)


@app.get(f'/{Config.BASE_URL}/{Config.API_VERSION}/info')
def get_info():
    return {
        'App': 'California Housing Prediction API',
        'Version': '1.0.0',
        'Time': datetime.now()
    }


# @app.post(f'/{Config.BASE_URL}/{Config.API_VERSION}/predict-housing-price')
# def predict_housing_price(request_body: dict):
#     print(request_body)
#     return {}



if __name__ == "__main__":
    uvicorn.run("main:app", host=Config.HOST, port=Config.PORT, reload=True)
