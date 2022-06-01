import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(alpha=0.5, l1_ratio=0.5):
    data = pd.read_csv("data/wine-quality.csv")
    labels = data.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(data, labels)
    exp_name = "ElasticNet_wine"
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, y_pred)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.log_artifact("data/wine-quality.csv")
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    for i in range(10):
        train(alpha=np.random.random(), l1_ratio=np.random.random())
