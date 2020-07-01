from progettobioinf.data_retrieval import *
from progettobioinf.first_elaboration import *
from progettobioinf.remove_uncorrellated_data import *


def test_execute():
    epi, lab = retrieve_epigenomes_labels("K562", 200)

    rate_features_samples(epi)
    nan_detection(epi)
    epi = knn_imputation(epi)
    check_class_balance(lab, "K562")
    epi = drop_constant_features(epi)
    epi = data_normalization(epi)

    uncorrelated = execute_pearson(epi, lab, 0.01)
    drop_features(epi, uncorrelated)

    uncorrelated = execute_spearman(epi, lab, 0.01)
    drop_features(epi, uncorrelated)

    uncorrelated = execute_mic(epi, lab, 0.95)
    drop_features(epi, uncorrelated)
