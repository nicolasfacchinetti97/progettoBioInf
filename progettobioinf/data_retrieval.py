from typing import Tuple

import numpy as np
import pandas as pd
from epigenomic_dataset import load_epigenomes
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from tensorflow.keras.utils import Sequence
from ucsc_genomes_downloader import Genome

from progettobioinf.initial_setup import *


def get_genome(assebly):
    return Genome(assembly)

def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]


def one_hot_encode(genome, data: pd.DataFrame, nucleotides: str = "actg") -> np.ndarray:
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=nucleotides,
        batch_size=1
    ))


def flat_one_hot_encode(genome, data: pd.DataFrame, window_size: int, nucleotides: str = "actg") -> np.ndarray:
    return one_hot_encode(genome, data, nucleotides).reshape(-1, window_size * 4).astype(int)


def to_dataframe(x: np.ndarray, window_size: int, nucleotides: str = "actg") -> pd.DataFrame:
    return pd.DataFrame(
        x,
        columns=[
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )


def retrieve_epigenomes_labels(cell_line, window_size):
    logging.info("Loading epigenomics data and labels...")
    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )

    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )

    # promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1)
    # enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)

    ## Epigenomes Dictionary
    epigenomes = {
        "promoters": promoters_epigenomes,
        "enhancers": enhancers_epigenomes
    }

    labels = {
        "promoters": promoters_labels,
        "enhancers": enhancers_labels
    }

    return epigenomes, labels


def retrieve_sequences(epigenomes, genome, window_size):
    logging.info("Loading sequences data...")

    # Sequences Dictionary
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data, window_size),
            window_size
        )
        for region, data in epigenomes.items()
    }
    return sequences


def get_sequence_holdout(train: np.ndarray, test: np.ndarray, bed: pd.DataFrame, labels: np.ndarray, genome,
                         batch_size=32) -> Tuple[Sequence, Sequence]:
    logging.info("Computing train sequence data...")
    train = MixedSequence(
        x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
        y=labels[train],
        batch_size=batch_size
    )
    logging.info("Computing test sequence data...")
    test = MixedSequence(
        x=BedSequence(genome, bed.iloc[test], batch_size=batch_size),
        y=labels[test],
        batch_size=batch_size
    )
    return (train, test)


def are_sequences_retrieved(cell_line):
    return (os.path.exists('csv/' + cell_line + '/sequence_promoters.csv') and
            os.path.exists('csv/' + cell_line + '/sequence_enhancers.csv'))


def are_epigenomics_retrieved(cell_line):
    return (os.path.exists('csv/' + cell_line + '/intial_promoters.csv') and
            os.path.exists('csv/' + cell_line + '/initial_enhancers.csv'))


def are_labels_retrieved(cell_line):
    return (os.path.exists('csv/' + cell_line + '/labels_promoters.csv') and
            os.path.exists('csv/' + cell_line + '/labels_enhancers.csv'))


def are_epigenomics_elaborated(cell_line):
    return (os.path.exists('csv/' + cell_line + '/elaborated_promoters.csv') and
            os.path.exists('csv/' + cell_line + '/elaborated_enhancers.csv'))


def are_data_visualized(cell_line):
    return os.path.exists('img/' + cell_line + '/pca_decomposition.png')


def are_data_retrieved(cell_line):
    return (are_sequences_retrieved(cell_line) and are_epigenomics_retrieved and are_labels_retrieved(cell_line))


def is_done_features_correlation(cell_line, n, top_number):
    statement = True
    for region in ["promoters", "enhancers"]:
        statement = (statement and os.path.exists(
            'img/{}/scatter_plot_most_{}_uncorrelated_{}.png'.format(cell_line, n, region)) and
                     os.path.exists('img/{}/scatter_plot_most_{}_correlated_{}.png'.format(cell_line, n, region)) and
                     os.path.exists('img/{}/top_{}_different_features_{}.png'.format(cell_line, top_number, region)) and
                     os.path.exists('img/{}/top_{}_different_tuples_{}.png'.format(cell_line, top_number, region)))
    return statement


def are_data_elaborated(cell_line, n, top_number):
    return (are_epigenomics_elaborated(cell_line) and is_done_features_correlation(cell_line, n, top_number))


def load_csv_retrieved_data(cell_line):
    sequences_promoters = pd.read_csv('csv/' + cell_line + '/sequence_promoters.csv', sep=',', index_col=0)
    sequences_enhancers = pd.read_csv('csv/' + cell_line + '/sequence_enhancers.csv', sep=',', index_col=0)
    sequences = {"promoters": sequences_promoters, "enhancers": sequences_enhancers}

    initial_promoters = pd.read_csv('csv/' + cell_line + '/initial_promoters.csv', sep=',')
    initial_promoters.set_index(["chrom", "chromStart", "chromEnd", "strand"], inplace=True, drop=True)
    initial_enhancers = pd.read_csv('csv/' + cell_line + '/initial_enhancers.csv', sep=',')
    initial_enhancers.set_index(["chrom", "chromStart", "chromEnd", "strand"], inplace=True, drop=True)
    epigenomes = {"promoters": initial_promoters, "enhancers": initial_enhancers}

    labels_promoters = pd.read_csv('csv/' + cell_line + '/labels_promoters.csv', sep=',')
    labels_promoters.set_index(["chrom", "chromStart", "chromEnd", "strand"], inplace=True, drop=True)
    labels_enhancers = pd.read_csv('csv/' + cell_line + '/labels_enhancers.csv', sep=',')
    labels_enhancers.set_index(["chrom", "chromStart", "chromEnd", "strand"], inplace=True, drop=True)
    labels = {"promoters": labels_promoters, "enhancers": labels_enhancers}

    return epigenomes, labels, sequences


def load_csv_elaborated_data(cell_line):
    elaborated_promoters = pd.read_csv('csv/' + cell_line + '/elaborated_promoters.csv', sep=',')
    elaborated_promoters.set_index(["chrom", "chromStart", "chromEnd", "strand"], inplace=True, drop=True)
    elaborated_enhancers = pd.read_csv('csv/' + cell_line + '/elaborated_enhancers.csv', sep=',')
    elaborated_enhancers.set_index(["chrom", "chromStart", "chromEnd", "strand"], inplace=True, drop=True)
    epigenomes = {"promoters": elaborated_promoters, "enhancers": elaborated_enhancers}

    return epigenomes
