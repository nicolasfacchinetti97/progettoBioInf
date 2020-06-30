from progettobioinf import *
from ucsc_genomes_downloader import Genome


def get_sequences_data():
    genome = Genome("hg19")
    epi, lab = retrieve_epigenomes_labels("MCF-7", 200)
    seq = retrieve_sequences(epi, genome, 200):