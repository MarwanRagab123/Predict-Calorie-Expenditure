import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

class Preprocessor:
    def __init__(self, path):
        self.path = path
        try:
            self.train_data = pd.read_csv(self.path)
        except FileNotFoundError:
            print(f"File not found: {self.path}")
            self.train_data = pd.DataFrame()

    def copy_data(self):
        """Return a copy of the training data."""
        return self.train_data.copy()

    def show_data(self, df):
        """Print the first 5 rows of the DataFrame."""
        print(df.head())

    def split_X_y(self, df):
        """Split DataFrame into features X and target y."""
        X = df.drop(columns=["Calories"])
        y = df["Calories"]
        return X, y

    def preprocess(self, df):
        """Preprocess DataFrame: drop 'id', encode categoricals, scale numericals."""
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        # Handle categorical data
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        # Scale numerical data
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df

if __name__ == "__main__":
    data_path = os.path.join("data", "train.csv")
    preprocessor = Preprocessor(data_path)
    if not preprocessor.train_data.empty:
        X, y = preprocessor.split_X_y(preprocessor.train_data)
        X = preprocessor.preprocess(X)
        print(X.head())
    else:
        print("No data to process.")