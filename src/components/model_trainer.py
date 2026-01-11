import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,  
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grids for tuning (used with RandomizedSearchCV)
            param_grids = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },
                "XGB Regressor": {
                    "n_estimators": [100, 200, 300, 400],
                    "learning_rate": [0.01, 0.1, 0.2, 0.7],
                    "max_depth": [3, 5, 7, 9],
                },
                "CatBoost Regressor": {
                    "iterations": [100, 200],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1, 0.2, 0.7],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                },
            }

            # Tune models with RandomizedSearchCV where a grid is provided
            for name, model in list(models.items()):
                params = param_grids.get(name)
                if params:
                    logging.info(f"Tuning {name} with RandomizedSearchCV")
                    try:
                        # compute total combinations in the param grid
                        total_combinations = 1
                        for v in params.values():
                            total_combinations *= len(v)

                        n_iter_search = min(10, total_combinations)
                        if n_iter_search <= 0:
                            logging.warning(f"No hyperparameter combinations for {name}, skipping tuning")
                            continue

                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=params,
                            n_iter=n_iter_search,
                            cv=3,
                            scoring="r2",
                            n_jobs=-1,
                            random_state=42,
                        )
                        search.fit(X_train, y_train)
                        models[name] = search.best_estimator_
                        logging.info(f"Best params for {name}: {search.best_params_} (used n_iter={n_iter_search})")
                    except Exception as e:
                        logging.warning(f"Tuning failed for {name}, using default estimator: {e}")

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test, models=models,param=param_grids)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with r2 score: {best_model_score}")  

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)