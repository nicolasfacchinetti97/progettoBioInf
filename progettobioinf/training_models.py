import logging
import os

import compress_json
import numpy as np
import pandas as pd
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tqdm.auto import tqdm

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def training_the_models(holdouts, splits, models, kwargs, X, y, cell_line, task):
    get_results(holdouts, splits, models, kwargs, X, y, cell_line, task)


def __report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    integer_metrics = accuracy_score, balanced_accuracy_score
    float_metrics = roc_auc_score, average_precision_score
    results1 = {
        sanitize_ml_labels(metric.__name__): metric(y_true, np.round(y_pred))
        for metric in integer_metrics
    }
    results2 = {
        sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
        for metric in float_metrics
    }
    return {
        **results1,
        **results2
    }


def __precomputed(results, model: str, holdout: int) -> bool:
    df = pd.DataFrame(results)
    if df.empty:
        return False
    return (
            (df.model == model) &
            (df.holdout == holdout)
    ).any()


def get_results(holdouts, splits, models, kwargs, X, y, cell_line, task):
    results = []
    for i, (train, test) in tqdm(enumerate(holdouts.split(X, y)), total=splits, desc="Computing holdouts",
                                 dynamic_ncols=True):
        for model, params in tqdm(zip(models, kwargs), total=len(models), desc="Training models", leave=False,
                                  dynamic_ncols=True):
            model_name = (
                model.__class__.__name__
                if model.__class__.__name__ != "Sequential"
                else model.name
            )
            if __precomputed(results, model_name, i):
                continue
            logging.info("Fit " + model_name)
            model.fit(X[train], y[train], **params)

            logging.info("Run Type: Train - " + model_name)
            results.append({
                "model": model_name,
                "run_type": "train",
                "holdout": i,
                **__report(y[train], model.predict(X[train]))
            })
            logging.info("Append results train " + model_name)

            logging.info("Run Type: Test - " + model_name)
            logging.info("Append results test " + model_name)
            results.append({
                "model": model_name,
                "run_type": "test",
                "holdout": i,
                **__report(y[test], model.predict(X[test]))
            })

            logging.info("Add results to Json --> results_" + task + ".json")
            compress_json.local_dump(results, "json/" + cell_line + "/results_" + task + ".json")

    return results
