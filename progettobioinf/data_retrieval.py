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