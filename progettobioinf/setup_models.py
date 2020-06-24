from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping

from classifier import *
from data_retrieval import *
from initial_setup import *


def setup_sequence_models(shape_value, n_holdouts, bed, labels, genome):
    models = []
    kwargs = []

    logging.info("Setup MLP")
    mlp = get_mpl_seq(shape_value)
    models.append(mlp)

    logging.info("Setup FFNN Network")
    ffnn = get_ffnn_seq(shape_value)
    models.append(ffnn)

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

    rand = get_random_forest_classifier()
    models.append(rand)
    kwargs.append({})

    logging.info("Setup Multi-Layer Perceptron 2")
    mlp2 = get_mlp_epi2(shape_value)
    models.append(mlp2)
    kwargs.append(dict(
        epochs=600,
        batch_size=512,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=30)
        ]
    ))

    logging.info("Setup Feed-Forward Neural Network")
    ffnn = get_ffnn(shape_value)
    models.append(ffnn)
    kwargs.append(dict(
        epochs=600,
        batch_size=512,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=30)
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
