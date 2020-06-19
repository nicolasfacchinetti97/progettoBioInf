import logging

from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping

from classifier import *
from data_retrieval import *

logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def setup_sequence_models(shape_value, n_holdouts, bed, labels, genome):
    models = []
    kwargs = []

    logging.info("Setup Convolutional Neural Network")
    cnn = get_cnn(shape_value)
    models.append(cnn)

    # computing the sequence data for the holdouts
    logging.info("Computing the holdouts...")
    holdouts = get_holdouts(n_holdouts)
    sequence_holdouts = []
    for train_index, test_index in holdouts.split(bed, labels):
        sequence_holdouts.append(get_sequence_holdout(train_index, test_index, bed, labels, genome))
    return models, sequence_holdouts


def setup_tabular_models(shape_value, n_holdouts, epigenomes, labels):
    models = []
    kwargs = []

    # logging.info("Setup KNN Classifier")
    # knn_classifier = get_knn_classifier()
    # models.append(knn_classifier)
    # kwargs.append({})

    """
    logging.info("Setup Decision Tree Classifier")
    decision_tree_classifier = get_decision_tree_classifier()
    models.append(decision_tree_classifier)
    kwargs.append({})

    logging.info("Setup Random Forest Classifier")
    random_forest_classifier = get_random_forest_classifier()
    models.append(random_forest_classifier)
    kwargs.append({})

    logging.info("Setup Single-Layer Perceptron")
    slp = get_slp(shape_value)
    models.append(slp)
    kwargs.append(dict(
        epochs=600,
        batch_size=512,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50)
        ]
    ))

    """

    logging.info("Setup Multi-Layer Perceptron")
    mlp = get_mlp(shape_value)
    models.append(mlp)
    kwargs.append(dict(
        epochs=200,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50)
        ]
    ))

    logging.info("Setup Feed-Forward Neural Network")
    ffnn = get_ffnn(shape_value)
    models.append(ffnn)
    kwargs.append(dict(
        epochs=200,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50)
        ]
    ))

    holdouts = get_holdouts(n_holdouts)
    logging.info("Computing the holdouts...")
    tabular_holdouts = []
    for train, test in holdouts.split(epigenomes, labels):
        train_data = {
            "epigenomes": epigenomes[train],
            "labels": labels[train]
        }
        test_data = {
            "epigenomes": epigenomes[test],
            "labels": labels[test]
        }
        tabular_holdouts.append([train_data, test_data])
    return models, kwargs, tabular_holdouts


def get_holdouts(splits):
    # TODO scegli quante volte fare gli holdout (n_splits = ?)
    holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)
    return holdouts
