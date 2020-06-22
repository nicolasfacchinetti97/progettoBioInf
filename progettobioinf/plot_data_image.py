import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

from initial_setup import *


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
        plt.savefig('img/{}/scatter_plot_most_{}_correlated_{}.png'.format(cell_line, n, region))
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
        plt.savefig('img/{}/scatter_plot_most_{}_uncorrelated_{}.png'.format(cell_line, n, region))
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
        plt.savefig('img/{}/top_{}_different_features_{}.png'.format(cell_line, top_number, region))
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
        plt.savefig('img/{}/top_{}_different_tuples_{}.png'.format(cell_line, top_number, region))
        logging.info('Top different tuples saved')
