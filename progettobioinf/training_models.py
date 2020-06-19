<<<<<<< HEAD
import logging
=======
from typing import Tuple
>>>>>>> 0226a802d901d949755a27e88bef59b23b9d7032

import compress_json
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tqdm.auto import tqdm

from initial_setup import *
from setup_models import *




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


<<<<<<< HEAD
def training_tabular_models(holdouts, models, kwargs, cell_line, task):
    results = []
    logging.info("Number of holdouts: {}".format(len(holdouts)))
    for i, (train, test) in enumerate(holdouts):
        print("{}\n***\n{}".format(train, test))
=======
def get_sequence_holdout(train: np.ndarray, test: np.ndarray, bed: pd.DataFrame, labels: np.ndarray, genome,
                         batch_size=1024) -> Tuple[Sequence, Sequence]:
    logging.info("Computing train sequence data...")
    train = MixedSequence(
        x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
        y=labels[train],
        batch_size=batch_size
    )
    logging.info("Computing test sequence data...")
    test = MixedSequence(
        x=BedSequence(genome, bed.iloc[test], batch_size=batch_size),
        y=labels[test],
        batch_size=batch_size
    )
    return (train, test)


def training_tabular_models(holdouts, splits, models, kwargs, X, y, cell_line, task):
    results = []
    for i, (train, test) in tqdm(enumerate(holdouts.split(X, y)), total=splits, desc="Computing holdouts",
                                 dynamic_ncols=True):
>>>>>>> 0226a802d901d949755a27e88bef59b23b9d7032
        for model, params in tqdm(zip(models, kwargs), total=len(models), desc="Training models", leave=False,
                                  dynamic_ncols=True):
            model_name = (
                model.__class__.__name__
                if model.__class__.__name__ != "Sequential"
                else model.name
            )
            if __precomputed(results, model_name, i):
                continue
            logging.info("Training model {} holdout {}".format(model_name, i))
            
            model.fit(train["epigenomes"], train["labels"], **params)

            results.append({
                "model": model_name,
                "run_type": "train",
                "holdout": i,
                **__report(train["labels"], model.predict(train["epigenomes"]))
            })
            results.append({
                "model": model_name,
                "run_type": "test",
                "holdout": i,
                **__report(test["labels"], model.predict(test["epigenomes"]))
            })

            logging.info("Add results {} to Json --> results_tabular_{}.json".format(model_name, task))
            compress_json.local_dump(results, "json/" + cell_line + "/results_tabular_" + task + ".json")
    return results

def training_sequence_models(models, holdouts, cell_line, task):    
    results = []
<<<<<<< HEAD
    logging.info("Number of holdouts: {}".format(len(holdouts)))
    for i, (train, test) in enumerate(holdouts):
=======

    for i, (train_index, test_index) in tqdm(enumerate(holdouts.split(bed, labels)), total=splits,
                                             desc="Computing holdouts", dynamic_ncols=True):
        train, test = get_sequence_holdout(train_index, test_index, bed, labels, genome)
>>>>>>> 0226a802d901d949755a27e88bef59b23b9d7032
        for model in tqdm(models, total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            if __precomputed(results, model.name, i):
                continue
            logging.info("Training model {} holdout {}".format(model.name, i))
            history = model.fit(
                train,
                steps_per_epoch=train.steps_per_epoch,
                validation_data=test,
                validation_steps=test.steps_per_epoch,
                epochs=200,
                shuffle=True,
                verbose=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=50),
                ]
            ).history
            scores = pd.DataFrame(history).iloc[-1].to_dict()
            results.append({
                "model": model.name,
                "run_type": "train",
                "holdout": i,
                **{
                    key: value
                    for key, value in scores.items()
                    if not key.startswith("val_")
                }
            })
            results.append({
                "model": model.name,
                "run_type": "test",
                "holdout": i,
                **{
                    key[4:]: value
                    for key, value in scores.items()
                    if key.startswith("val_")
                }
            })
            logging.info("Add results {} to Json --> results_sequence_{}.json".format(model.name, task))
            compress_json.local_dump(results, "json/" + cell_line + "/results_sequence_" + task + ".json")
    return results
