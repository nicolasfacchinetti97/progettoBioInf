from tqdm.auto import tqdm # A simple loading bar
import matplotlib.pyplot as plt # A standard plotting library
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
from cache_decorator import Cache
from glob import glob 
import seaborn as sns

cell_line = "MCF-7"
window_size = 200                      # the width of the window of nucleotides we are going to take in consideration

# use the epigenomic_dataset library to get the data prepared by ohters!
# in this way no need of downlaod data from Encode, labels from Fantom5 and querying the bigwig files since is already done!
from epigenomic_dataset import load_epigenomes

promoters_epigenomes, promoters_labels = load_epigenomes(
    cell_line = cell_line,
    dataset = "fantom",
    regions = "promoters",
    window_size = window_size
)

enhancers_epigenomes, enhancers_labels = load_epigenomes(
    cell_line = cell_line,
    dataset = "fantom",
    regions = "enhancers",
    window_size = window_size
)

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

# now we have the epigenetic data, lets get also the sequence ones
from ucsc_genomes_downloader import Genome      # genome browser of the university of california, santa cruz

genome = Genome(assembly)

#we can now extract the bed coordinates from the promoters and enachers that we
# previously donwloaded
def to_bed(data:pd.DataFrame)->pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]

display(to_bed(epigenomes["promoters"])[:5])