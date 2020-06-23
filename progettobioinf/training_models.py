import compress_json
from keras.callbacks import EarlyStopping
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tqdm.auto import tqdm

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


def training_tabular_models(holdouts, models, kwargs, cell_line, task):
    results = []
    logging.info("Number of holdouts: {}".format(len(holdouts)))
    for i, (train, test) in enumerate(holdouts):
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
    logging.info("Number of holdouts: {}".format(len(holdouts)))
    for i, (train, test) in enumerate(holdouts):
        for model in tqdm(models, total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            if __precomputed(results, model.name, i):
                continue

            logging.info("Training model {} holdout {}".format(model.name, i))
            print("steps per epoch {}\nvalidation steps {}".format(train.steps_per_epoch, test.steps_per_epoch))
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
