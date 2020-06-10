from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
import numpy as np
import pandas as pd
from keras_bed_sequence import BedSequence
import logging

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

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

    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1) 
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)
    
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
    # Genome from UCSC
    genome = Genome(assembly)

    # Sequences Dictionary
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data, window_size),
            window_size
        )
        for region, data in epigenomes.items()  # TODO check this line
    }
    return