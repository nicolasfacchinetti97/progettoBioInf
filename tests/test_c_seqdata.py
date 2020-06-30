
from progettobioinf.data_retrieval import get_genome, retrieve_epigenomes_labels, retrieve_sequences


def test_get_sequences_data():
    genome = get_genome("hg19")
    epi, lab = retrieve_epigenomes_labels("MCF-7", 200)
    seq = retrieve_sequences(epi, genome, 200)