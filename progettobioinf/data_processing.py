from progettobioinf.data_retrieval import *
from progettobioinf.data_visualization import *
from progettobioinf.initial_setup import *


# Data Retrieval
def data_retrieval(cell_line, genome, window_size):
    # retrieve epigenomic, labels and sequences data
    epigenomes, labels = retrieve_epigenomes_labels(cell_line, window_size)
    sequences = retrieve_sequences(epigenomes, genome, window_size)

    if os.path.exists('csv/' + cell_line + '/sequence_promoters.csv'):
        logging.info('sequence_promoters already exists')
    else:
        sequences['promoters'].to_csv('csv/' + cell_line + '/sequence_promoters.csv', sep=',')
    if os.path.exists('csv/' + cell_line + '/sequence_enhancers.csv'):
        logging.info('sequence_enhancers already exists')
    else:
        sequences['enhancers'].to_csv('csv/' + cell_line + '/sequence_enhancers.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/initial_promoters.csv'):
        logging.info('Initial_promoters already exists')
    else:
        epigenomes['promoters'].to_csv('csv/' + cell_line + '/initial_promoters.csv', sep=',')
    if os.path.exists('csv/' + cell_line + '/initial_enhancers.csv'):
        logging.info('Initial_enhancers already exists')
    else:
        epigenomes['enhancers'].to_csv('csv/' + cell_line + '/initial_enhancers.csv', sep=',')

    if os.path.exists('csv/' + cell_line + '/labels_enhancers.csv'):
        logging.info('labels_enhancers already exists')
    else:
        labels['enhancers'].to_csv('csv/' + cell_line + '/labels_enhancers.csv', sep=',')
    if os.path.exists('csv/' + cell_line + '/labels_promoters.csv'):
        logging.info('labels_promoters already exists')
    else:
        labels['promoters'].to_csv('csv/' + cell_line + '/labels_promoters.csv', sep=',')

    return epigenomes, labels, sequences


# Step 2. Data elaboration
def data_elaboration(epigenomes, labels, cell_line, number_tuples, top_number):
    logging.info("Starting Data Elaboration")

    if are_epigenomics_elaborated(cell_line):
        logging.info("The epigenomics data already elaborated! Load from .csv files...")
        epigenomes = load_csv_elaborated_data(cell_line)
    else:
        epigenomes = elaborate_epigenomics_data(epigenomes, labels, cell_line)

    # ====================== Features correalations ======================
    if is_done_features_correlation(cell_line, number_tuples, top_number):
        logging.info("Feature correlations image alredy done... skip step")
    else:
        f = open("log/info.txt", "a+")
        f.write("=" * 80)
        f.write("\n")
        f.write("Checking Features Correlations... \n")
        f.close()
        do_features_correlations(epigenomes, labels, cell_line, number_tuples, top_number)

    # ====================== Features selection automatica ======================
    # result_boruta = start_feature_selection(epigenomes, labels)

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
