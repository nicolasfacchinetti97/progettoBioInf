from progettobioinf.data_processing import *

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

        logging.info("Check if data are already elaborated/visualized")
        if are_data_elaborated(cell_line) and are_sequence_data_elaborated(cell_line) and are_data_already_visualized(
                cell_line):

            logging.info('Data already elaborated and visualized. Skipping Data Elaboration/Visualization and load '
                         'csv files.')

            logging.info("Loading sequences data.")
            elaborated_sequences_promoters = pd.read_csv('csv/' + cell_line + '/sequence_promoters.csv', sep=',')
            elaborated_sequences_enhancers = pd.read_csv('csv/' + cell_line + '/sequence_enhancers.csv', sep=',')
            sequences = {"promoters": elaborated_sequences_promoters, "enhancers": elaborated_sequences_enhancers}

            logging.info("Loading epigenomes data.")
            elaborated_promoters = pd.read_csv('csv/' + cell_line + '/elaborated_promoters.csv', sep=',')
            elaborated_enhancers = pd.read_csv('csv/' + cell_line + '/elaborated_enhancers.csv', sep=',')
            epigenomes = {"promoters": elaborated_promoters, "enhancers": elaborated_enhancers}

            logging.info("Loading labels data.")
            elaborated_promoters_labels = pd.read_csv('csv/' + cell_line + '/initial_labels_promoters.csv', sep=',')
            elaborated_enhancers_labels = pd.read_csv('csv/' + cell_line + '/initial_labels_enhancers.csv', sep=',')
            labels = {"promoters": elaborated_promoters_labels, "enhancers": elaborated_enhancers_labels}

        else:
            logging.info("Data are not elaborated/visualized.")

            # Step 1. Data Retrieval
            logging.info('Step 1. Data Retrieval')
            epigenomes, labels, sequences = dataRetrieval(cell_line, assembly, window_size)

            # Step 2. Data Elaboration
            logging.info('Step 2. Data Elaboration')
            epigenomes = dataElaboration(epigenomes, labels, cell_line)

            # Step 3. Data Visualization
            logging.info('Step 3. Data Visualization')
            data_visualization(epigenomes, labels, sequences, cell_line)

        # TODO
        # Step 4. Holdout

        # Step 5. Fit models (Prediction of Active Enhancers/Promoters)

        # Step 6. Measuring Performance

        # Step 7. Statistical Test (Wilcoxon)

        logging.info('Exiting cell_line' + cell_line)


if __name__ == '__main__':
    main()
