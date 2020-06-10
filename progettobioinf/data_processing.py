from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome

from data_retrieval import *
from data_visualization import *
from initial_setup import *

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

    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1) 
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)
    
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

    ## Sequences Dictionary
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data, window_size),
            window_size
        )
        for region, data in epigenomes.items()  # TODO check this line
    }

    # TODO remove this 2 lines of code...
    logging.info('Saving epigenomes csv')
    save_dictionary_as_csv('epigenomes.csv', epigenomes)

    # TODO aggiungi questo codice in una funzione sotto, refactorizza questa parte e aggiungi le loggate quando crea i vari csv
    if os.path.exists('csv/' + cell_line + '/sequence_promoters.csv'):
        logging.info('sequence_promoters already exists')
    else:
        epigenomes['promoters'].to_csv('csv/' + cell_line + '/sequence_promoters.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/sequence_enhancers.csv'):
        logging.info('sequence_enhancers already exists')
    else:
        epigenomes['enhancers'].to_csv('csv/' + cell_line + '/sequence_enhancers.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/initial_promoters.csv'):
        logging.info('Initial_promoters already exists')
    else:
        epigenomes['promoters'].to_csv('csv/' + cell_line + '/initial_promoters.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/initial_enhancers.csv'):
        logging.info('Initial_enhancers already exists')
    else:
        epigenomes['enhancers'].to_csv('csv/' + cell_line + '/initial_enhancers.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/initial_labels_enhancers.csv'):
        logging.info('labels_enhancers already exists')
    else:
        labels['enhancers'].to_csv('csv/' + cell_line + '/initial_labels_enhancers.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/initial_labels_promoters.csv'):
        logging.info('labels_promoters already exists')
    else:
        labels['promoters'].to_csv('csv/' + cell_line + '/initial_labels_promoters.csv', sep=',')

    return epigenomes, labels, sequences


# Step 2. Data elaboration
def dataElaboration(epigenomes, labels, cell_line):
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
    check_class_balance(labels, cell_line)

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
    logging.info("Sorting the obtained scores")
    scores = {
        region: sorted(score, key=lambda x: np.abs(x[0]), reverse=True)
        for region, score in scores.items()
    }

    ## Scatter plot (most correlated touples)
    logging.info("Scatter plot - detect most n correlated touples")
    detect_most_n_correlated_touples(epigenomes, scores, 3, labels, cell_line)

    ## Scatter plot (most uncorrelated touples)
    logging.info("Scatter plot - detect most n uncorrelated touples")
    detect_most_n_uncorrelated_touples(epigenomes, scores, 3, labels, cell_line)

    # Features distributions
    logging.info("Getting top n different features")
    get_top_n_different_features(epigenomes, labels, 5, cell_line)

    logging.info("Getting top n different touples")
    get_top_n_different_tuples(epigenomes, 5, cell_line)

    start_feature_selection(epigenomes, labels)

    for region, x in epigenomes.items():
        logging.info('Saving epigenomes elaborated csv (' + region + ')')
        # np.savetxt('epigenomes_elaborated_np_' + region + '.csv', epigenomes[region], delimiter=',')
        epigenomes[region].to_csv('csv/' + cell_line + '/elaborated_' + region + '.csv', sep=',')

    logging.info("Exiting data elaboration")
    return epigenomes


# Step 3. Data Visualization
def data_visualization(epigenomes, labels, sequences, cell_line):
    logging.info("Starting data visualization")

    # epigenomes["promoters"].drop(epigenomes["promoters"].columns[[0, 3]], axis=1,
    #                              inplace=True)
    #
    # epigenomes["enhancers"].drop(epigenomes["enhancers"].columns[[0, 3]], axis=1,
    #                              inplace=True)
    #
    # labels["promoters"].drop(labels["promoters"].columns[[0, 3]], axis=1,
    #                          inplace=True)
    # labels["enhancers"].drop(labels["enhancers"].columns[[0, 3]], axis=1,
    #                          inplace=True)

    xs, ys, titles, colors = prepare_data(epigenomes, labels, sequences)

    ## PCA
    visualization_PCA(xs, ys, titles, colors, cell_line)

    logging.info("Exiting Data Visualization")


def load_data_from_csv(cell_line):
    logging.info("Loading sequences data.")
    elaborated_sequences_promoters = pd.read_csv('csv/' + cell_line + '/sequence_promoters.csv', sep=',')
    elaborated_sequences_enhancers = pd.read_csv('csv/' + cell_line + '/sequence_enhancers.csv', sep=',')
    sequences = {"promoters": elaborated_sequences_promoters, "enhancers": elaborated_sequences_enhancers}

    logging.info("Loading epigenomes data.")
    elaborated_promoters = pd.read_csv('csv/' + cell_line + '/elaborated_promoters.csv', sep=',')
    elaborated_enhancers = pd.read_csv('csv/' + cell_line + '/elaborated_enhancers.csv', sep=',')
    epigenomes = {"promoters": elaborated_promoters, "enhancers": elaborated_enhancers}

    logging.info("Loading labels data.")
    fields = [cell_line]
    initial_labels_promoters = pd.read_csv('csv/' + cell_line + '/initial_labels_promoters.csv', sep=',',
                                           usecols=fields)
    initial_labels_enhancers = pd.read_csv('csv/' + cell_line + '/initial_labels_enhancers.csv', sep=',',
                                           usecols=fields)
    labels = {"promoters": initial_labels_promoters, "enhancers": initial_labels_enhancers}

    return epigenomes, labels, sequences

