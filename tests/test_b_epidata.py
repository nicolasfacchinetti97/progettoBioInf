from progettobioinf.data_retrieval import *


def test_get_epigenomics_data():
    epi, lab = retrieve_epigenomes_labels("MCF-7", 200)
