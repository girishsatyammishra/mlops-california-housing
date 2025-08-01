from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd


class DataProcessor:

    @staticmethod
    def load_housing_data():
        # # load dataset from sklearn dataser
        # california = fetch_california_housing()
        # df = pd.DataFrame(california.data, columns=california.feature_names)
        # df['MedHouseValue'] = california.target

        df = pd.read_csv('../resources/data/housing.csv')

        return df

    @staticmethod
    def split_train_test_dataset(df, target_column='median_house_value'):
        # features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # split dataset into train n test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def preprocess_dataset(df):
        df = df.copy()  # to avoid chnages on original df

        # Handle missing values for 'total_bedrooms'
        if 'total_bedrooms' in df.columns:
            df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

        # Feature engineering
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        df['population_per_household'] = df['population'] / df['households']

        # categorical encoding
        if 'ocean_proximity' in df.columns:
            if len(df) > 1:
                df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
            else:
                df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)

        return df

