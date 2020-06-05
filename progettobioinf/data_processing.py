from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome

from progettobioinf.data_retrieval import *
from progettobioinf.data_visualization import *
from progettobioinf.initial_setup import *
import logging
logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

# Data Retrieval
def dataRetrieval(cell_line, assembly, window_size):

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
        for region, data in epigenomes.items() # TODO check this line
    }

    # print(epigenomes["promoters"][:2])
    # print(epigenomes["enhancers"][:2])
    # print(sequences["promoters"][:2])
    # print(sequences["enhancers"][:2])

    # TODO remove this code...
    logging.info('Saving epigenomes csv')
    save_dictionary_as_csv('epigenomes.csv', epigenomes)

    return epigenomes, labels, sequences

# Step 2. Data elaboration
def dataElaboration(epigenomes, labels):
    logging.info("Starting Data Elaboration")

    ## Rate between features and samples
    logging.info("Rate features samples")
    rate_features_samples(epigenomes)

    ## NaN detection
    logging.info("NaN Detection")
    nan_detection(epigenomes)

    ## KNN imputation
    logging.info("KNN imputation")
    epigenomes = knn_imputation(epigenomes)

    ## Class Balance
    logging.info("Checking class balance")
    check_class_balance(labels)

    ## Drop constant features
    logging.info("Dropping constant features")
    epigenomes = drop_constant_features(epigenomes)

    ## Z-Scoring
    logging.info("Data normalization")
    epigenomes = data_normalization(epigenomes)

    p_value_threshold = 0.01
    correlation_threshold = 0.05

    uncorrelated = {
        region: set()
        for region in epigenomes
    }

    ## Pearson
    logging.info("Executing Pearson Test")
    execute_pearson(epigenomes, labels, p_value_threshold, uncorrelated)

    ## Spearman
    logging.info("Executing Spearman Test")
    execute_spearman(epigenomes, labels, p_value_threshold, uncorrelated)

    ## MIC
    logging.info("Executing Mic Test")
    execute_mic(epigenomes, labels, correlation_threshold, uncorrelated)

    ## Drop features uncorrelated with output
    logging.info("Dropping features uncorrelated with output")
    epigenomes = drop_features(epigenomes, uncorrelated)

    extremely_correlated = {
        region: set()
        for region in epigenomes
    }

    scores = {
        region: []
        for region in epigenomes
    }

    # Features correlations
    logging.info("Checking features correlations")
    check_features_correlations(epigenomes, scores, p_value_threshold, correlation_threshold, extremely_correlated)

    # Sort the obtained scores
    scores = {
        region: sorted(score, key=lambda x: np.abs(x[0]), reverse=True)
        for region, score in scores.items()
    }

    ## Scatter plot (most correlated touples)
    logging.info("Scatter plot - detect most n correlated touples")
    detect_most_n_correlated_touples(epigenomes, scores, 3, labels)

    ## Scatter plot (most uncorrelated touples)
    logging.info("Scatter plot - detect most n uncorrelated touples")
    detect_most_n_uncorrelated_touples(epigenomes, scores, 3, labels)

    # Features distributions
    logging.info("Getting top n different features")
    get_top_n_different_features(epigenomes, labels, 5)

    logging.info("Getting top n different tuples")
    get_top_n_different_tuples(epigenomes, 5)

    # Features selection
    logging.info("Starting feature selection - Boruta")
    epigenomes = start_feature_selection(epigenomes, labels)

    # Last step
    # TODO check this code...
    logging.info("Saving elaboration state")
    save_elaboration_state()

    # TODO remove this code
    logging.info('Saving epigenomes elaborated csv')
    save_dictionary_as_csv('epigenomes_elaborated.csv', epigenomes)

    for region, x in epigenomes.items():
        logging.info('Saving epigenomes elaborated csv (' + region + ')')
        np.savetxt('epigenomes_elaborated_np_' + region + '.csv', epigenomes[region], delimiter=',')

    for region, y in labels.items():
        logging.info('Saving labels (' + region + ')')
        np.savetxt('labels_elaborated_' + region + '.csv', labels[region], delimiter=',')

    logging.info("Exiting data elaboration")
    return epigenomes

# Step 3. Data Visualization
def data_visualization(epigenomes, labels, sequences):
    logging.info("Starting data visualization")
    # Data visualization
    visualization_data = prepare_data(epigenomes, labels, sequences)
    xs = visualization_data[0]
    ys = visualization_data[1]
    titles = visualization_data[2]
    colors = visualization_data[3]

    ## PCA
    # visualization_PCA(xs, ys, titles, colors)
    # TODO fix pca error: ValueError: could not convert string to float: 'promoters'


    ## TSNE
    #visualization_TSNE(xs, ys, titles, colors)
    # TODO tsne-cuda funziona solo su linux?
    # controlla errore su TSNE: OSError: no file with expected extension

    logging.info("Exiting Data Visualization")