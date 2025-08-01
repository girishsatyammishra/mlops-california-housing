from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .base_model import BaseModel
import mlflow


class DecisionTreeModel(BaseModel):

    def __init__(self, max_depth=None, random_state=42):
        super().__init__(model_name="DecisionTreeRegressor")
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self.features = []

    def train_n_evaluate(self, X_train, y_train,  X_test, y_test):

        # log features
        self.features = X_train.columns.tolist()

        # mlflow - log hyperparameters
        with mlflow.start_run(run_name=self.model_name):
            self.model.fit(X_train, y_train)

            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("random_state", self.random_state)

            decision_tree_mse, decision_tree_r2, decision_tree_rmse, decision_tree_mae = self.evaluate(X_test, y_test)

            mlflow.sklearn.log_model(self.model, artifact_path=self.model_name)

            return decision_tree_mse, decision_tree_r2, decision_tree_rmse, decision_tree_mae

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
