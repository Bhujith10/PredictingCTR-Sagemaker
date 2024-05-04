"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile
import os
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import f1_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
        
    logger.info('list of files ',os.listdir())
    
    # file_path = "xgboost-model"
    # if os.path.exists(file_path):
    #     print(f"The file {file_path} exists.")
    #     # Additional code to load the model or perform other actions
    # else:
    #     print(f"The file {file_path} does not exist.")
    logger.debug("Loading xgboost model.")
    model = xgboost.Booster() 
    model.load_model("xgboost-model") 
    # model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    print(df.head())

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df = df.drop(df.columns[0], axis=1)
    x_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = list(model.predict(x_test))
    predictions = [1 if x>=0.5 else 0 for x in predictions] 

    logger.debug("Calculating f1 score")
    f1_score = f1_score(y_test, predictions)
    
    logger.info(f"F1 Score {f1_score}")
    
    report_dict = {
        "classification_metrics": {
            "f1_score": {
                "value": f1_score
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with f1 score: %f", f1_score)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
