import os

# to suppress the annoying logging of tensorlow         DEVE STARE IN CIMA!
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
# to suppress the annoy log of keras "Using Tensorflow backend."
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

from data_processing import *
from training_models import *
from results import *
from ucsc_genomes_downloader import Genome


def main():
    logging.info('Started')

    cell_lines = ["K562"]
    assembly = "hg19"
    window_size = 200

    logging.info("Loading the genome {} for further elaboration...".format(assembly))
    genome = Genome(assembly)

    for cell_line in cell_lines:
        logging.info('Cell Line: ' + cell_line)

        # Step 0. Initial Setup
        logging.info('Step 0. Initial Setup')
        initial_setup(cell_line)

        # Step 1. Data Retrieval
        logging.info('Step 1. Data Retrieval')
        if are_data_retrieved(cell_line):
            logging.info('Data already retrieved! Load from .csv files...')
            epigenomes, labels, sequences = load_csv_retrieved_data(cell_line)
        else:
            epigenomes, labels, sequences = dataRetrieval(cell_line, genome, window_size)

        # Step 2. Data Elaboration
        logging.info('Step 2. Data Elaboration')
        top_number = 5  # parameters for features correlation
        number_tuples = 3
        if are_data_elaborated(cell_line, number_tuples, top_number):
            logging.info('All the elaborations done! Load from .csv files...')
            epigenomes = load_csv_elaborated_data(cell_line)
        else:
            epigenomes = dataElaboration(epigenomes, labels, cell_line, number_tuples, top_number)

        # Step 3. Data Visualization
        logging.info('Step 3. Data Visualization')
        if are_data_visualized(cell_line):
            logging.info("Data already visualized! Skip the step...")
        else:
            data_visualization(epigenomes, labels, sequences, cell_line)

        # Step 4. Training the models
        logging.info("Step 4. Training the models")

        # TODO aggiungere meta modelli per setup parametri

        for region, x in epigenomes.items():
            if os.path.exists('json/' + cell_line + '/results_tabular_' + region + ".json"):
                logging.info("Tabular results for " + region + " ok! Skip...")

            else:
                logging.info("Step 4.1 Training Tabular Data " + region)
                n_holdouts = 2

                converted_labels = labels[region].values.ravel()
                logging.info("labels shape: {}".format(converted_labels.shape))
                converted_epigenomes = epigenomes[region].values
                logging.info("Shape of epigenomics data for {}: {}".format(region, converted_epigenomes.shape))
                logging.info("Setup models for Tabular Data: " + region)
                data_shape = converted_epigenomes.shape[1]
                models, kwargs, holdouts = setup_tabular_models(data_shape, n_holdouts, converted_epigenomes, converted_labels)
                results = training_tabular_models(holdouts, models, kwargs, cell_line, region)

        for region, x in epigenomes.items():
            if os.path.exists('json/' + cell_line + '/results_sequence_' + region + ".json"):
                logging.info("Sequence results for " + region + " ok! Skip...")

            else:
                logging.info("Step 4.2 Training Sequence Data " + region)
                # TODO scegliere quanti holdouts fare
                n_holdouts = 2
                
                converted_labels = labels[region].values.ravel()
                logging.info("labels shape: {}".format(converted_labels.shape))
                bed = epigenomes[region].reset_index()[epigenomes[region].index.names]   # get the bed data (index data frame)
                logging.info("Shape of epigenomics data for {}: {}".format(region, bed.shape))
                logging.info("Setup models for Sequence Data: " + region)
                
                models, holdouts = setup_sequence_models(window_size, n_holdouts, bed, converted_labels, genome)
                training_sequence_models(models, holdouts, cell_line, region)

        # Step 5. Results and statistical tests
        logging.info("Step 5. Results and statistical tests")
        for region, x in epigenomes.items():
            logging.info("Results and statistical test for region: {}".format(region))
            if os.path.exists('json/' + cell_line + '/results_tabular_' + region + ".json"):
                df = pd.read_json("json/" + cell_line + "/results_tabular_" + region + ".json")
                path_barplots_cell_line = "img/" + cell_line + "/results_" + region + "_{feature}"
                df = df.drop(columns=["holdout"])
                generate_barplots(df, path_barplots_cell_line, region)
                get_wilcoxon(df)
            else:
                logging.error("Results for region " + region + " are not available.")

        logging.info('Exiting cell_line' + cell_line)


if __name__ == '__main__':
    main()
