from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np


class BaseModel(ABC):

    def __init__(self, model_name="BaseModel"):
        self.model = None
        self.model_name = model_name

    @abstractmethod
    def train_n_evaluate(self, X_train, y_train, X_test_scaled, y_test):
        raise NotImplementedError

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):

        if self.model is None:
            raise ValueError("Model has not been initialized.")

        # Predictions
        y_pred = self.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # MLflow logging
        # mlflow.log_param("model_name", self.model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Save the model as an MLflow artifact
        mlflow.sklearn.log_model(self.model, artifact_path=f'{self.model_name}')

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R Squared Score: {r2:.4f}")

        return mse, r2, rmse, mae
