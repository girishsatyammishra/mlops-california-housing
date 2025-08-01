import pandas as pd
import numpy as np
from app.data_processor import DataProcessor
from app.model import LinearRegressionModel, DecisionTreeModel, modelManager
from .best_model_finder import best_model_finder


class HousingPricePredictionService:

    linear_regression_model: LinearRegressionModel = None
    decision_tree_model: DecisionTreeModel = None

    def __init__(self, force=True):
        """
        load pretrained models if already saved
        or
        initiate and train
        """
        existing_lr_model = modelManager.get_model('LinearRegression')
        existing_dt_model = modelManager.get_model('DecisionTreeRegressor')

        if existing_lr_model is not None and existing_dt_model is not None and not force:
            print('Loading pretrained saved models')
            self.linear_regression_model = existing_lr_model
            self.decision_tree_model = existing_dt_model
        else:
            print('Initiating new models and training')
            self.linear_regression_model, self.decision_tree_model = self.init_models()
            modelManager.save_model('LinearRegression', self.linear_regression_model)
            modelManager.save_model('DecisionTreeRegressor', self.decision_tree_model)

    def init_models(self):
        # load dataset df
        df = DataProcessor.load_housing_data()

        # print for data check
        print(df.head(5))

        # data preprocessing
        preprocessed_df = DataProcessor.preprocess_dataset(df)

        # split dataset
        X_train, X_test, y_train, y_test = DataProcessor.split_train_test_dataset(preprocessed_df, 'median_house_value')

        # Initialize models
        linear_model = LinearRegressionModel()
        decision_tree_model = DecisionTreeModel(max_depth=6)

        # Train n evaluate Linear Regression model
        print("Linear Regression Performance:")
        linear_regression_mse, linear_regression_r2, linear_regression_rmse, linear_regression_mae = linear_model.train_n_evaluate(X_train, y_train, X_test, y_test)

        # Train n evaluate Decision Tree Regression model
        print("\nDecision Tree Regression Performance:")
        decision_tree_mse, decision_tree_r2, decision_tree_rmse, decision_tree_mae = decision_tree_model.train_n_evaluate(X_train, y_train, X_test, y_test)

        # # Train models
        # linear_model.train(X_train_scaled, y_train)
        # decision_tree_model.train(X_train, y_train)
        #
        # # print model evaluation
        # print("Linear Regression Performance:")
        # linear_regression_mse, linear_regression_r2 = linear_model.evaluate(X_test_scaled, y_test)
        #
        # print("\nDecision Tree Regression Performance:")
        # decision_tree_mse, decision_tree_r2 = decision_tree_model.evaluate(X_test, y_test)

        return linear_model, decision_tree_model

    def predict(self, input):
        # load to df and preprocess
        df = pd.DataFrame([input.dict()])
        df = DataProcessor.preprocess_dataset(df)

        # get best performing model
        best_performing_model, best_performing_model_name = best_model_finder.get_best_model()

        # order feature columns in same order of training features
        feature_columns = modelManager.get_model(best_performing_model_name).features
        print(feature_columns)
        # df = df.reindex(columns=feature_columns, fill_value=None)
        df = df.reindex(columns=feature_columns, fill_value=0)

        prediction = best_performing_model.predict(df)[0]

        return {"predicted_price": round(prediction, 2)}

    def predict_linear_regression(self, X):
        return self.linear_regression_model.predict(X)

    def predict_decision_tree_regression(self, X):
        return self.decision_tree_model.predict(X)


housingPricePredictionService = HousingPricePredictionService()
