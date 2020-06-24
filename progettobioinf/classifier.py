from multiprocessing import cpu_count

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential


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
        n_estimators=400,               # numero di tree
        criterion="gini",
        max_depth=30,
        random_state=42,
        class_weight="balanced",
        n_jobs=cpu_count()
    )
    return random_forest_classifier


def get_mlp_epi(shape_value):
    mlp = Sequential([
        Input(shape=(shape_value,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "MLP1")

    mlp.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]
    )

    return mlp

def get_mlp_epi2(shape_value):
    mlp = Sequential([
        Input(shape=(shape_value,)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "MLP2")

    mlp.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]
    )

    return mlp


def get_ffnn(shape_value):
    ffnn = Sequential([
        Input(shape=(shape_value,)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
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
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]

    )

    return ffnn


def get_mpl_seq(shape_value):
    mlp = Sequential([
        Input(shape=(200, 4)),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(32, activation= "relu"),
        Dense(1, activation="sigmoid")
    ], "MLP")

    mlp.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]
    )

    return mlp

def get_ffnn_seq(shape_value):
    ffnn = Sequential([
        Input(shape=(200, 4)),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "FFNN")

    ffnn.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]
    )
    return ffnn    

def get_cnn(shape_value):
    shape_v = (shape_value, 4)

    cnn = Sequential([
        Input(shape=shape_v),
        Reshape((*shape_v, 1)),
        Conv2D(64, kernel_size=(10, 2), activation="relu"),
        Dropout(0.3),
        Conv2D(32, kernel_size=(10, 2), strides=(2, 1), activation="relu"),
        Conv2D(32, kernel_size=(10, 1), activation="relu"),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "CNN")

    cnn.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]
    )

    return cnn
