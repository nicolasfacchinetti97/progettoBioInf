import os

# to suppress the annoying logging of tensorlow         DEVE STARE IN CIMA!
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from data_processing import *
from setup_models import *
from training_models import *
from results import *

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def main():
    logging.info('Started')

    cell_lines = ["K562"]
    assembly = "hg19"
    window_size = 200

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
            epigenomes, labels, sequences = dataRetrieval(cell_line, assembly, window_size)

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
                logging.info("Results " + region + " ok!")

            else:
                logging.info("Step 4.1 Training Tabular Data " + region)

                converted_labels = labels[region].values.ravel()

                logging.info("labels shape: {}".format(converted_labels.shape))
                converted_epigenomes = epigenomes[region].values
                logging.info("Shape of epigenomics data for {}: {}".format(region, converted_epigenomes.shape))
                logging.info("Setup models for Tabular Data: " + region)
                # list of models, args for training, indeces train/test, num splits
                models, kwargs, holdouts, splits = setup_tabular_models(converted_epigenomes.shape[1])
                training_tabular_models(holdouts, splits, models, kwargs, converted_epigenomes, converted_labels,
                                        cell_line, region)

                # TODO uncomment this
                # logging.info("Step 4.2 Training Sequence Data" + region)
                # bed = epigenomes[region].reset_index()[epigenomes[region].index.names]  # TODO che cosa fa?
                # logging.info("Shape of epigenomics data for {}: {}".format(region, sequences[region].shape))
                # logging.info("Setup models for Sequence Data: " + region)
                # models, kwargs, holdouts, splits = setup_sequence_models(sequences[region].shape)
                # training_sequence_models(holdouts, splits, models, kwargs, bed, converted_labels, genome, cell_line, region)

        # Step 5. Results and statistical tests
        # TODO
        logging.info("Step 5. Results and statistical tests")
        # results = get_results(holdouts, splits, models, kwargs, X, y)
        # results_df = convert_results_to_dataframe(results)
        # save_results_df_to_csv(results_df)
        # save_barplots_to_png()
        # TODO aggiungi test statistici e plot dei risultati

        for region, x in epigenomes.items():
            logging.info("Statistical test for " + region + " region")
            if os.path.exists('json/' + cell_line + '/results_tabular_' + region + ".json"):
                df = pd.read_json("json/" + cell_line + "/results_tabular_" + region + ".json")
                get_wilcoxon(df)
            else:
                logging.error("Results for region " + region + " are not available.")

        logging.info('Exiting cell_line' + cell_line)


if __name__ == '__main__':
    main()
