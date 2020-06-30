from progettobioinf.classifier import *

def test_models():
    get_decision_tree_classifier()
    get_ffnn(200)
    get_ffnn_seq(200)
    get_mlp_epi2(200)
    get_mpl_seq(200)
    get_cnn(200)