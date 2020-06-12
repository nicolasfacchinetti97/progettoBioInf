import logging
from typing import Tuple

import compress_json
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tensorflow.keras.utils import Sequence
from tqdm.auto import tqdm

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


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


def __get_sequence_holdouts(train: np.ndarray, test: np.ndarray, bed: pd.DataFrame, labels: np.ndarray, genome,
                            batch_size=1024) -> Tuple[Sequence, Sequence]:
    return (
        MixedSequence(
            x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
            y=labels[train],
            batch_size=batch_size
        ),
        MixedSequence(
            x=BedSequence(genome, bed.iloc[test], batch_size=batch_size),
            y=labels[test],
            batch_size=batch_size
        )
    )


def training_tabular_models(holdouts, splits, models, kwargs, X, y, cell_line, task):
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

            results.append({
                "model": model_name,
                "run_type": "train",
                "holdout": i,
                **__report(y[train], model.predict(X[train]))
            })
            results.append({
                "model": model_name,
                "run_type": "test",
                "holdout": i,
                **__report(y[test], model.predict(X[test]))
            })

            logging.info("Add results {} to Json --> results_tabular_{}.json".format(model_name, task))
            compress_json.local_dump(results, "json/" + cell_line + "/results_tabular_" + task + ".json")
    return results


def training_sequence_models(holdouts, splits, models, kwargs, bed, labels, genome, cell_line, task):
    results = []

    for i, (train_index, test_index) in tqdm(enumerate(holdouts.split(bed, labels)), total=splits,
                                             desc="Computing holdouts", dynamic_ncols=True):
        train, test = __get_sequence_holdouts(train_index, test_index, bed, labels, genome)
        for model in tqdm(models, total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            if __precomputed(results, model.name, i):
                continue
            history = model.fit(
                train,
                steps_per_epoch=train.steps_per_epoch,
                validation_data=test,
                validation_steps=test.steps_per_epoch,
                epochs=1000,
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
