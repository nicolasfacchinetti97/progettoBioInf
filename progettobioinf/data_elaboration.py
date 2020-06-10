import logging
import os
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from boruta import BorutaPy
from minepy import MINE
from scipy.stats import entropy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm  # A simple loading bar

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


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


# Correlation with output

## Linear Correlation (Pearson)
def execute_pearson(epigenomes, labels, p_value_threshold, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = pearsonr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                logging.info("p_value > p_value_threshold")
                uncorrelated[region].add(column)


## Monotonic Correlation (Spearman)
def execute_spearman(epigenomes, labels, p_value_threshold, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = spearmanr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                logging.info("Adding column to uncorrelated region")
                uncorrelated[region].add(column)


## Non-Linear Correlation (MIC)
def execute_mic(epigenomes, labels, correlation_threshold, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True,
                           leave=False):
            mine = MINE()
            mine.compute_score(x[column].values.ravel(), labels[region].values.ravel())
            score = mine.mic()
            if score < correlation_threshold:
                logging.info("score < correlation_threshold")
            else:
                logging.info("score > correlation_threshold")
                logging.info("Removing column from uncorrelated region")
                uncorrelated[region].remove(column)


### Drop features uncorrelated with output
def drop_features(epigenomes, uncorrelated):
    for region, x in epigenomes.items():
        epigenomes[region] = x.drop(columns=[
            col
            for col in uncorrelated[region]
            if col in x.columns
        ])
    return epigenomes


# Features correlations
def check_features_correlations(epigenomes, scores, p_value_threshold, correlation_threshold, extremely_correlated):
    for region, x in epigenomes.items():
        for i, column in tqdm(
                enumerate(x.columns),
                total=len(x.columns), desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            for feature in x.columns[i + 1:]:
                correlation, p_value = pearsonr(x[column].values.ravel(), x[feature].values.ravel())
                correlation = np.abs(correlation)
                scores[region].append((correlation, column, feature))
                if p_value < p_value_threshold and correlation > correlation_threshold:
                    if entropy(x[column]) > entropy(x[feature]):
                        extremely_correlated[region].add(feature)
                    else:
                        extremely_correlated[region].add(column)


# Most "n" correlated touples
def detect_most_n_correlated_touples(epigenomes, scores, n, labels, cell_line):
    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][:n]))
        columns = list(set(firsts + seconds))
        logging.info(f"Most correlated features from {region} epigenomes")
        sns.pairplot(pd.concat([
            x[columns],
            labels[region],
        ], axis=1), hue=labels[region].columns[0])
        plt.savefig('img/' + cell_line + '/scatter_plot_correlated_' + region + ".png")
        logging.info('Scatter plot correlated saved')


# Most "n" uncorrelated touples
def detect_most_n_uncorrelated_touples(epigenomes, scores, n, labels, cell_line):
    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][-n:]))
        columns = list(set(firsts + seconds))
        logging.info(f"Least correlated features from {region} epigenomes")
        sns.pairplot(pd.concat([
            x[columns],
            labels[region],
        ], axis=1), hue=labels[region].columns[0])
        plt.savefig('img/' + cell_line + '/scatter_plot_uncorrelated_' + region + ".png")
        logging.info('Scatter plot uncorrelated saved')


# Features distributions
def __get_top_most_different(dist, n: int):
    return np.argsort(-np.mean(dist, axis=1).flatten())[:n]


def get_top_n_different_features(epigenomes, labels, top_number, cell_line):
    for region, x in epigenomes.items():
        dist = euclidean_distances(x.T)
        most_distance_columns_indices = __get_top_most_different(dist, top_number)
        columns = x.columns[most_distance_columns_indices]
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        logging.info(f"Top {top_number} different features from {region}.")
        for column, axis in zip(columns, axes.flatten()):
            head, tail = x[column].quantile([0.05, 0.95]).values.ravel()

            mask = ((x[column] < tail) & (x[column] > head)).values

            cleared_x = x[column][mask]
            cleared_y = labels[region].values.ravel()[mask]

            cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
            cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

            axis.set_title(column)

        fig.tight_layout()
        plt.savefig('img/' + cell_line + '/top_' + str(top_number) + '_different_features_' + region + '.png')
        logging.info('Top different features saved')


def __get_top_most_different_tuples(dist, n: int):
    return list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:n]


def get_top_n_different_tuples(epigenomes, top_number, cell_line):
    for region, x in epigenomes.items():
        dist = euclidean_distances(x.T)
        dist = np.triu(dist)
        tuples = __get_top_most_different_tuples(dist, top_number)
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        logging.info(f"Top {top_number} different tuples of features from {region}.")
        for (i, j), axis in zip(tuples, axes.flatten()):
            column_i = x.columns[i]
            column_j = x.columns[j]
            for column in (column_i, column_j):
                head, tail = x[column].quantile([0.05, 0.95]).values.ravel()
                mask = ((x[column] < tail) & (x[column] > head)).values
                x[column][mask].hist(ax=axis, bins=20, alpha=0.5)
            axis.set_title(f"{column_i} and {column_j}")
        fig.tight_layout()
        plt.savefig('img/' + cell_line + '/top_' + str(top_number) + '_different_tuples_' + region + '.png')
        logging.info('Top different tuples saved')


def get_features_filter(X: pd.DataFrame, y: pd.DataFrame) -> BorutaPy:
    boruta_selector = BorutaPy(
        RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5),
        n_estimators='auto',
        verbose=2,
        alpha=0.05,  # p_value
        max_iter=10,  # In practice one would run at least 100-200 times
        random_state=42
    )
    boruta_selector.fit(X.values, y.values.ravel())
    return boruta_selector


def start_feature_selection(epigenomes, labels):
    filtered_epigenomes = {
        region: get_features_filter(
            X=x,
            y=labels[region]
        ).transform(x.values)

        for region, x in tqdm(
            epigenomes.items(),
            desc="Running Baruta Feature estimation"
        )
    }
    return filtered_epigenomes


def pca(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(x)


def are_data_elaborated(cell_line):
    return are_enhancers_elaborated(cell_line) and are_promoters_elaborated(
        cell_line) and are_labels_enhancers_elaborated(cell_line) and are_labels_promoters_elaborated(cell_line)


def are_sequence_data_elaborated(cell_line):
    return are_sequence_enhancers_elaborated(cell_line) and are_sequence_promoters_elaborated(cell_line)


def are_sequence_promoters_elaborated(cell_line):
    return os.path.exists('csv/' + cell_line + '/sequence_promoters.csv')


def are_sequence_enhancers_elaborated(cell_line):
    return os.path.exists('csv/' + cell_line + '/sequence_enhancers.csv')


def are_promoters_elaborated(cell_line):
    return os.path.exists('csv/' + cell_line + '/elaborated_promoters.csv')


def are_enhancers_elaborated(cell_line):
    return os.path.exists('csv/' + cell_line + '/elaborated_enhancers.csv')


def are_labels_promoters_elaborated(cell_line):
    return os.path.exists('csv/' + cell_line + '/initial_labels_promoters.csv')


def are_labels_enhancers_elaborated(cell_line):
    return os.path.exists('csv/' + cell_line + '/initial_labels_enhancers.csv')
