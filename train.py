from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import os
import joblib

class Trainer(Preprocessor):
    def __init__(self, path):
        super().__init__(path)

    def split_data_into_train_val(self, df):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, val_df

    def train_and_evaluate(self, models):
        df = self.copy_data()
        X, y = self.split_X_y(df)
        X = self.preprocess(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "DecisionTree": {
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10]
            },
            "XGBRegressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }

        best_models = {}  # to store the final models

        for name, model in models.items():
            print(f"\nTraining {name}...")
            if name in params:
                grid = GridSearchCV(
                    model, params[name], 
                    cv=5, scoring="neg_mean_squared_error", n_jobs=-1
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                print(f"Best params for {name}: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
                best_model = model

            preds = best_model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            print(f"{name} MSE: {mse:.4f}")

            best_models[name] = best_model  # save the final models

        return best_models


if __name__ == "__main__":
    data_path = os.path.join("data", "train.csv")
    trainer = Trainer(data_path)
    models = {
        "DecisionTree": DecisionTreeRegressor(),
        "XGBRegressor": XGBRegressor()
    }

    if not trainer.train_data.empty:
        best_models = trainer.train_and_evaluate(models)

        # save models
        os.makedirs("model", exist_ok=True)
        for name, model in best_models.items():
            joblib.dump(model, f"model/{name}_best_model.pkl")
            print(f" Saved {name} model at model/{name}_best_model.pkl")
    else:
        print("No data to process.")
