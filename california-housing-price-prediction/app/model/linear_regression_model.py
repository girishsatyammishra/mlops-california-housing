from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from .base_model import BaseModel
import mlflow


class LinearRegressionModel(BaseModel):

    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None):
        super().__init__(model_name="LinearRegression")
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs
        )
        self.features = []

    def train_n_evaluate(self, X_train, y_train, X_test, y_test):

        # Standardize for Linear Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # log features
        self.features = X_train.columns.tolist()

        # mlflow - log hyperparameters
        with mlflow.start_run(run_name=self.model_name):
            self.model.fit(X_train_scaled, y_train)

            mlflow.log_param("fit_intercept", self.fit_intercept)
            mlflow.log_param("copy_X", self.copy_X)
            mlflow.log_param("n_jobs", self.n_jobs if self.n_jobs is not None else "None")

            linear_regression_mse, linear_regression_r2, linear_regression_rmse, linear_regression_mae = self.evaluate(X_test_scaled, y_test)

            mlflow.sklearn.log_model(self.model, artifact_path=self.model_name)

            return linear_regression_mse, linear_regression_r2, linear_regression_rmse, linear_regression_mae

    # def evaluate(self, X_test, y_test):
    #     # Predictions
    #     y_pred = self.predict(X_test)
    #
    #     # Evaluation
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)
    #     print(f"Mean Squared Error (MSE): {mse:.4f}")
    #     print(f"R Squared Score: {r2:.4f}")
    #
    #     return mse, r2
    #
    # def predict(self, X):
    #     return self.model.predict(X)
