import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

from initial_setup import *




# Rate between features and samples
def rate_features_samples(epigenomes):
    for region, x in epigenomes.items():
        logging.info(
            f"The rate between features and samples for {region} data is: {x.shape[0] / x.shape[1]}"
        )
        logging.info("=" * 80)


# NaN Detection
def nan_detection(epigenomes):
    for region, x in epigenomes.items():
        logging.info("\n".join((
            f"Nan values report for {region} data:",
            f"In the document there are {x.isna().values.sum()} NaN values out of {x.values.size} values.",
            f"The sample (row) with most values has {x.isna().sum(axis=0).max()} NaN values out of {x.shape[1]} values.",
            f"The feature (column) with most values has {x.isna().sum().max()} NaN values out of {x.shape[0]} values."
        )))
        logging.info("=" * 80)


# KNN imputer
def __knn_imputer(df: pd.DataFrame, neighbours: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        KNNImputer(n_neighbors=neighbours).fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )


def knn_imputation(epigenomes):
    for region, x in epigenomes.items():
        epigenomes[region] = __knn_imputer(x)
    return epigenomes


# Class Balance
def check_class_balance(labels, cell_line):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for axis, (region, y) in zip(axes.ravel(), labels.items()):
        y.hist(ax=axis, bins=3)
        axis.set_title(f"Classes count in {region}")
    fig.savefig('img/' + cell_line + '/class_balance.png')
    logging.info('class balance img saved')


# Constant Features
def __drop_const_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame without constant features."""
    return df.loc[:, (df != df.iloc[0]).any()]


def drop_constant_features(epigenomes):
    for region, x in epigenomes.items():
        result = __drop_const_features(x)
        if x.shape[1] != result.shape[1]:
            logging.info(f"Features in {region} were constant and had to be dropped!")
            epigenomes[region] = result
        else:
            logging.info(f"No constant features were found in {region}!")
    return epigenomes


# Z-scoring
def __robust_zscoring(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )


def data_normalization(epigenomes):
    epigenomes = {
        region: __robust_zscoring(x)
        for region, x in epigenomes.items()
    }
    logging.info("Data normalized")
    return epigenomes
