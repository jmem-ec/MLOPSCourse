from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
#from build_features import GetData
import os
import json
import joblib
import sys
import argparse
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import datetime

MODEL_DIR = "models"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_DIR)
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
folder_name = CURRENT_TIME_STAMP
MAIN_PATH = os.path.join(MODEL_PATH, folder_name)

class TrainEvaluate:
    def __init__(self):
        #self.get_data = GetData()
        self.filename = "model_rf.pkl"

    def evaluation_metrics(self, act, pred):
        self.r2_score = r2_score(act, pred)
        self.mse = mean_squared_error(act, pred)
        self.rmse = np.sqrt(mean_squared_error(act, pred))
        return self.r2_score, self.mse, self.rmse

    def model_eval(self):
        try:
            print("'train_evaluate' function started")
            #self.config = self.get_data.read_params(config_path)
            self.test_data = "data/processed/test.csv"
            self.train_data = "data/processed/train.csv"
            self.model_dir = "models"
            self.target_col = "line_item_value"
            
            print("train data read successfully-->path: "+self.train_data)
            self.train = pd.read_csv(self.train_data, sep=",")
            print("train data read successfully")
            self.test = pd.read_csv(self.test_data, sep=",")
            print("test data read successfully")
            
            
            print("model training started")
            self.criterion = "mae"
            self.max_deapth = 10
            self.min_sample_leaf = 2
            self.n_estimators = 80
            self.min_sample_split = 8
            self.oob_score = True
            
            self.x_train, self.x_test = self.train.drop(
                self.target_col, axis=1), self.test.drop(self.target_col, axis=1)
            self.y_train, self.y_test = self.train[self.target_col], self.test[self.target_col]



            rf = RandomForestRegressor()
            
            rf.fit(self.x_train, self.y_train)
            
            distributions = { 
                "n_estimators": [5,10,20,40,80],
                "criterion": ["absolute_error", 'friedman_mse', 'poisson', 'squared_error'],
                "max_depth": [2,5,10],
                "min_samples_split": [2,4,8,12],
                "min_samples_leaf": [2,4,8,10]
            }

            RCV = RandomizedSearchCV(
                estimator=rf,
                param_distributions=distributions,
                n_iter=3,
                scoring="r2",
                cv=5,
                verbose=5,
                random_state=42,
                n_jobs=-1,
                return_train_score=True
            )
            rf1 = RCV.fit(self.x_train, self.y_train)
            
            print(RCV.best_score_)
            
            
                # Feature importances
            # self.rf_feature_imp = pd.DataFrame(
            #     rf.feature_importances_, index=self.x_train.columns, columns=['Feature_importance'])

            # self.rf_feature_imp.sort_values(
            #     by='Feature_importance', ascending=False, inplace=True)

            # print(round(self.rf_feature_imp, 3))
                
            # self.rf1_feature_imp['Feature_importance'] = self.rf1_feature_imp[self.rf1_feature_imp['Feature_importance'] > 0.001]

            # self.rf1_feature_imp = self.rf1_feature_imp[self.rf1_feature_imp['Feature_importance'].notna()]
            # self.features_by_rf = self.rf1_feature_imp.index

            y_pred = rf1.predict(self.x_test)
            print("Model Trained on RandomizedSearchCV successfully")
            
            (r2, mse, rmse) = self.evaluation_metrics(self.y_test, y_pred)
            print(r2*100, mse, rmse)

            # normalized_rmse=rmse/(63770.43-1121)
            # print(f"normalized rmse::{normalized_rmse}")

            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(MAIN_PATH,exist_ok=True)
            self.model_path = os.path.join(MAIN_PATH,self.filename)
            joblib.dump(rf1, self.model_path)

            scores_file = "reports/scores.json"
            params_file = "reports/params.json"

            with open(scores_file, "w") as f:
                scores = {
                    "rmse": rmse,
                    "r2 score": r2*100,
                    "mse": mse,
                    "train_score": rf.score(self.x_train, self.y_train),
                    "test_score": rf.score(self.x_test, self.y_test)
                }
                json.dump(scores, f, indent=4)
            print("scores written to file")
            
            with open(params_file, "w") as f:
                params = {
                    "best params": RCV.best_params_,
                    "criterion": self.criterion,
                    "n_estimators": self.n_estimators,
                    "max_deapth": self.max_deapth,
                    "min_sample_leaf": self.min_sample_leaf,
                    "min_sample_split": self.min_sample_split,
                    "oob_score": self.oob_score
                }
                json.dump(params, f, indent=4)

        except Exception as e:
            print("Exception occured in 'train_evaluate' function"+str(e))
            print("train_evaluate function reported error in the function")
            raise Exception(e, sys) from e
        
if __name__ == "__main__":
    TrainEvaluate().model_eval()
