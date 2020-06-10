import logging

from keras_tqdm import TQDMNotebookCallback as ktqdm
from tensorflow.keras.callbacks import EarlyStopping

from classifier import *
from sklearn.model_selection import StratifiedShuffleSplit

logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def setup_model_cnn(shape_value):
    models = []
    kwargs = []

    logging.info("Setup Convolutional Neural Network")
    cnn = get_cnn(shape_value)
    models.append(cnn)
    kwargs.append(dict(
        epochs=10,
        batch_size=16,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]

#epochs=1000,
#        shuffle=True,
#        verbose=False,
#        callbacks=[
#            EarlyStopping(monitor="val_loss", mode="min", patience=50),
#        ]
    ))

    holdouts, splits = __preparing_the_holdouts()
    return models, kwargs, holdouts, splits

def setup_models_ffnn(shape_value):
    models = []
    kwargs = []

    # logging.info("Setup KNN Classifier")
    # knn_classifier = get_knn_classifier()
    # models.append(knn_classifier)
    # kwargs.append({})

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
        epochs=800,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    ))

    logging.info("Setup Multi-Layer Perceptron")
    mlp = get_mlp(shape_value)
    models.append(mlp)
    kwargs.append(dict(
        epochs=800,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    ))

    logging.info("Setup Feed-Forward Neural Network")
    ffnn = get_ffnn(shape_value)
    models.append(ffnn)
    kwargs.append(dict(
        epochs=800,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    ))

    #logging.info("Setup Convolutional Neural Network")
    #cnn = get_cnn(shape_value)
    #models.append(cnn)
    #kwargs.append(dict)

    holdouts, splits = __preparing_the_holdouts()
    return models, kwargs, holdouts, splits


def __preparing_the_holdouts():
    # TODO scegli quante volte fare gli holdout (n_splits = ?)
    splits = 9
    holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)
    return holdouts, splits