from progettobioinf.data_retrieval import *

def test_create_log():
    create_log_file()

def test_init():
    initial_setup("K562")


def test_init_already_exists():
    initial_setup("K562")


def test_data_retrieval():
    are_data_retrieved("K562")


def test_data_visualized():
    are_data_visualized("K562")


def test_data_elaborated():
    are_data_elaborated("K562", 3, 5)
