from progettobioinf.data_processing import *


def test_execute():
    epi, lab = retrieve_epigenomes_labels("MCF-7", 200)
    uncorrelated = execute_pearson(epi, lab, 0.01)
    drop_features(epi, uncorrelated)

    uncorrelated = execute_spearman(epi, lab, 0.01)
    drop_features(epi, uncorrelated)

    uncorrelated = execute_mic(epi, lab, 0.95)
    drop_features(epi, uncorrelated)
