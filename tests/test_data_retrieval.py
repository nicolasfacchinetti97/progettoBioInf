from progettobioinf.data_retrieval import *

def test_retrieve():
    genome = get_genome("hg19")
    epi, lab = retrieve_epigenomes_labels("K562", 200)
    seq = retrieve_sequences(epi, genome, 200)

def test_are_sequences_retrieved():
    are_sequences_retrieved("K562")


def test_are_epigenomics_retrieved():
    are_epigenomics_retrieved("K562")


def test_are_labels_retrieved():
    are_labels_retrieved("K562")


def test_are_epigenomics_elaborated():
    are_epigenomics_elaborated("K562")


def test_are_data_visualized():
    are_data_visualized("K562")


def test_are_data_retrieved():
    are_data_retrieved("K562")
