from progettobioinf.data_processing import *
from progettobioinf.results import *
from progettobioinf.setup_models import *
from progettobioinf.training_models import *

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
            fields = [cell_line]
            initial_labels_promoters = pd.read_csv('csv/' + cell_line + '/initial_labels_promoters.csv', sep=',',
                                                   usecols=fields)
            initial_labels_enhancers = pd.read_csv('csv/' + cell_line + '/initial_labels_enhancers.csv', sep=',',
                                                   usecols=fields)
            labels = {"promoters": initial_labels_promoters, "enhancers": initial_labels_enhancers}

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

        # Step 4. Training the models
        logging.info("Step 5. Training the models")

        for region, x in epigenomes.items():
            if os.path.exists('json/' + cell_line + '/results_' + region + ".json"):
                logging.info("Results " + region + " ok!")

            else:
                logging.info("Step 5.1 Training " + region)
                logging.info("Dropping non-numeric column")
                epigenomes[region].drop('chrom', axis=1, inplace=True)
                epigenomes[region].drop('chromStart', axis=1, inplace=True)
                epigenomes[region].drop('chromEnd', axis=1, inplace=True)
                epigenomes[region].drop('strand', axis=1, inplace=True)

                X = epigenomes[region].to_numpy()
                y = labels[region].to_numpy().ravel()
                logging.info("X shape: " + ''.join(str(X.shape)))
                logging.info("y shape: " + ''.join(str(y.shape)))

                logging.info("Setup models: " + region)
                models, kwargs, holdouts, splits = setup_models(X.shape[1])
                training_the_models(holdouts, splits, models, kwargs, X, y, cell_line, region)

        # if os.path.exists('json/' + cell_line + '/results_promoters.json'):
        #     logging.info("Results promoters ok!")
        # else:
        #     logging.info("Step 5.1 Training Promoters")
        #     logging.info("Dropping non-numeric column")
        #     epigenomes["promoters"].drop('chrom', axis=1, inplace=True)
        #     epigenomes["promoters"].drop('chromStart', axis=1, inplace=True)
        #     epigenomes["promoters"].drop('chromEnd', axis=1, inplace=True)
        #     epigenomes["promoters"].drop('strand', axis=1, inplace=True)
        #
        #     X = epigenomes["promoters"].to_numpy()
        #     y = labels["promoters"].to_numpy().ravel()
        #     logging.info("X shape: " + ''.join(str(X.shape)))
        #     logging.info("y shape: " + ''.join(str(y.shape)))
        #
        #     logging.info("Setup models (promoters)")
        #     models, kwargs, holdouts, splits = setup_models(X.shape[1])
        #     training_the_models(holdouts, splits, models, kwargs, X, y, cell_line, "promoters")
        #
        # if os.path.exists('json/' + cell_line + '/results_enhancers.json'):
        #     logging.info("Results promoters ok!")
        # else:
        #     logging.info("Step 5.2 Training Enhancers")
        #     logging.info("Dropping non-numeric column")
        #     epigenomes["enhancers"].drop('chrom', axis=1, inplace=True)
        #     epigenomes["enhancers"].drop('chromStart', axis=1, inplace=True)
        #     epigenomes["enhancers"].drop('chromEnd', axis=1, inplace=True)
        #     epigenomes["enhancers"].drop('strand', axis=1, inplace=True)
        #
        #     X = epigenomes["enhancers"].to_numpy()
        #     y = labels["enhancers"].to_numpy().ravel()
        #     logging.info("X shape: " + ''.join(str(X.shape)))
        #     logging.info("y shape: " + ''.join(str(y.shape)))
        #
        #     logging.info("Setup models (enhancers)")
        #     models, kwargs, holdouts, splits = setup_models(X.shape[1])
        #     training_the_models(holdouts, splits, models, kwargs, X, y, cell_line, "enhancers")

        # Step 6. Results and statistical tests
        #TODO
        logging.info("TODO!!! Step 6. Results and statistical tests")
        # results = get_results(holdouts, splits, models, kwargs, X, y)
        # results_df = convert_results_to_dataframe(results)
        # save_results_df_to_csv(results_df)
        # save_barplots_to_png()

        # TODO aggiungi test statistici e plot dei risultati

        logging.info('Exiting cell_line' + cell_line)


if __name__ == '__main__':
    main()
