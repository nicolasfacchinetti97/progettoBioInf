import pandas as pd
import numpy as np

from keras_bed_sequence import BedSequence
from epigenomic_dataset import load_epigenomes      # get epigenetic data prepared
from ucsc_genomes_downloader import Genome          # genome browser of the university of california, santa cruz


# use to extract the bed coordinates from the promoters and enachers
# the bed format provide a way do define the data lines that are displayed in an annotation track
def to_bed(data:pd.DataFrame)->pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]

# convert to one hot encode 
def one_hot_encode(genome:Genome, data:pd.DataFrame, nucleotides:str="actg")->np.ndarray:
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=nucleotides,
        batch_size=1
    ))

# flat the one hot encode result
def flat_one_hot_encode(genome:Genome, data:pd.DataFrame, window_size:int, nucleotides:str="actg")->np.ndarray:
    return one_hot_encode(genome, data, nucleotides).reshape(-1, window_size*4).astype(int)

# to convert back the one-hot encoded nucleotides into DataFrame
def to_dataframe(x:np.ndarray, window_size:int, nucleotides:str="actg")->pd.DataFrame:
    return pd.DataFrame(
        x,
        columns = [
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )

# use the epigenomic_dataset library to get the data prepared by ohters!
# in this way no need of downlaod data from Encode and the labels from Fantom5 and querying the bigwig files since is already done!
def get_promoters_enhancers(cell_line, window_size):
    print("Get promoters data...")
    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line = cell_line,
        dataset = "fantom",
        regions = "promoters",
        window_size = window_size
    )
    print("Done!\nGet enhancers data...")
    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line = cell_line,
        dataset = "fantom",
        regions = "enhancers",
        window_size = window_size
    )
    print("Done!")
    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1) 
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)
    
    # wrap the data in a dictionary for more pratical use
    epigenomes = {
        "promoters": promoters_epigenomes,
        "enhancers": enhancers_epigenomes
    }
    labels = {
        "promoters": promoters_labels,
        "enhancers": enhancers_labels
    }
    return epigenomes, labels                   # return a tuple containing the data

def get_sequence_data(assembly, window_size, epigenomes):
    print("Get genome data...")
    genome = Genome(assembly)
    print("Done!\nGet sequence data...")
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data, window_size),
            window_size
        )
        for region, data in epigenomes.items()
    }
    print("Done!")
    return sequences
