from multiprocessing import cpu_count

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential


# def get_knn_classifier():
#     knn = KNeighborsClassifier(n_neighbors=5)
#     return knn


def get_decision_tree_classifier():
    decision_tree_classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=50,
        random_state=42,
        class_weight="balanced"
    )
    return decision_tree_classifier


def get_random_forest_classifier():
    random_forest_classifier = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_depth=30,
        random_state=42,
        class_weight="balanced",
        n_jobs=cpu_count()
    )
    return random_forest_classifier


def get_slp(shape_value):
    slp = Sequential([
        Input(shape=(shape_value,)),
        Dense(1, activation="sigmoid")
    ], "Perceptron")

    slp.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    return slp


def get_mlp(shape_value):
    mlp = Sequential([
        Input(shape=(shape_value,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "MLP")

    mlp.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    return mlp


def get_ffnn(shape_value):
    ffnn = Sequential([
        Input(shape=(shape_value,)),
        Dense(256, activation="relu"),
        Dense(128),
        BatchNormalization(),
        Activation("relu"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "FFNN")

    ffnn.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    return ffnn
