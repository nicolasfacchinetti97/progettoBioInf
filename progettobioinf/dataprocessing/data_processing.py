from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome

from progettobioinf.dataprocessing.data_retrieval import *
from progettobioinf.dataprocessing.data_elaboration import *

cell_line = "K562"
assembly = "hg19"
window_size = 200

# Step 1. Data Retrieval
# Step 2. Data Elaboration
# Step 3. Data Visualization

if __name__ == '__main__':

    # Step 1. Data Retrieval

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

    ## Epigenomes Dictionary
    epigenomes = {
    "promoters": promoters_epigenomes,
    "enhancers": enhancers_epigenomes
    }

    labels = {
    "promoters": promoters_labels,
    "enhancers": enhancers_labels
    }

    ## Genome from UCSC
    genome = Genome(assembly)

    genome.bed_to_sequence(to_bed(epigenomes["promoters"])[:2])

    ## Flat One Hot Encode
    flat_one_hot_encode(genome, epigenomes["promoters"][:2], window_size)

    ## Sequences Dictionary
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data, window_size),
            window_size
        )
        for region, data in epigenomes.items()
    }

    print(epigenomes["promoters"][:2])
    print(epigenomes["enhancers"][:2])
    print(sequences["promoters"][:2])
    print(sequences["enhancers"][:2])

    # Step 2. Data elaboration

    ## Rate between features and samples
    rate_features_samples(epigenomes)

    ## NaN detection
    nan_detection(epigenomes)

    ## KNN imputation
    knn_imputation(epigenomes)

    ## Class Balance
    check_class_balance(labels)

    ## Drop constant features
    drop_constant_features(epigenomes)

    ## Z-Scoring
    data_normalization(epigenomes)