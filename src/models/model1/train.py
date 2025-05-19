from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
#from build_features import GetData
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import datetime

MODEL_DIR = "models"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_DIR)
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
MAIN_PATH = os.path.join(MODEL_PATH, CURRENT_TIME_STAMP)

class TrainEvaluate:
    def __init__(self):
        #self.get_data = GetData()
        self.filename = "model_rf.pkl"
        self.config = {
            "train_path": "data/processed/train.csv",
            "test_path": "data/processed/test.csv",
            "target_col": "target",  # <-- reemplaza por el nombre real de tu variable objetivo
            "model_dir": MODEL_PATH,
            "scores_file": "reports/scores.json",
            "params_file": "reports/params.json",
            "search_params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "bootstrap": [True, False]
            },
            "search_config": {
                "n_iter": 10,
                "scoring": "r2",
                "cv": 5,
                "verbose": 1,
                "random_state": 42,
                "n_jobs": -1,
                "return_train_score": True
            }
        }

    def evaluation_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return r2, mse, rmse

    def model_eval(self):
        # Leer los datos
        train = pd.read_csv(self.config["train_path"])
        test = pd.read_csv(self.config["test_path"])
        
        X_train = train.drop(self.config["target_col"], axis=1)
        y_train = train[self.config["target_col"]]
        X_test = test.drop(self.config["target_col"], axis=1)
        y_test = test[self.config["target_col"]]

        # Entrenar el modelo base
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)

        # RandomizedSearchCV
        RCV = RandomizedSearchCV(
            estimator=rf,
            param_distributions=self.config["search_params"],
            **self.config["search_config"]
        )
        rf_best = RCV.fit(X_train, y_train)

        print("Best score:", RCV.best_score_)

        # Predicciones y métricas
        y_pred = rf_best.predict(X_test)
        r2, mse, rmse = self.evaluation_metrics(y_test, y_pred)
        print(f"R2: {r2*100:.2f}%, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Guardar el modelo
        os.makedirs(self.config["model_dir"], exist_ok=True)
        os.makedirs(MAIN_PATH, exist_ok=True)
        model_path = os.path.join(MAIN_PATH, self.filename)
        joblib.dump(rf_best, model_path)

        # Guardar métricas
        os.makedirs("reports", exist_ok=True)
        with open(self.config["scores_file"], "w") as f:
            scores = {
                "rmse": rmse,
                "r2 score (%)": r2 * 100,
                "mse": mse,
                "train_score": rf.score(X_train, y_train),
                "test_score": rf.score(X_test, y_test)
            }
            json.dump(scores, f, indent=4)

        # Guardar parámetros
        with open(self.config["params_file"], "w") as f:
            params = {
                "best params": RCV.best_params_,
            }
            json.dump(params, f, indent=4)


if __name__ == "__main__":
    trainer = TrainEvaluate()
    trainer.model_eval()
