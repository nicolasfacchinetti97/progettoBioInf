from progettobioinf.dataprocessing.utility import *
from progettobioinf.dataprocessing.data_processing import check_class_balance
from epigenomic_dataset import load_epigenomes

def test_create_folder():
    create_img_folder()

def test_save_img_plot():

    cell_line = "K562"
    window_size = 200

    ## Epigenomic
    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )

    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )

    labels = {
    "promoters": promoters_labels,
    "enhancers": enhancers_labels
    }

    check_class_balance(labels)