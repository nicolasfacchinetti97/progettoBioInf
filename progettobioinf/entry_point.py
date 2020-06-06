from progettobioinf.data_processing import *
import logging
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
        initial_data = initial_setup(cell_line)

        # Step 1. Data Retrieval
        logging.info('Step 1. Data Retrieval')
        epigenomes, labels, sequences = dataRetrieval(cell_line, assembly, window_size)

        # Step 2. Data Elaboration
        logging.info('Step 2. Data elaboration')
        if are_data_elaborated(cell_line):
            logging.info('Data already elaborated. Skipping Data Elaboration and load csv files.')
            # enhancers = np.loadtxt('epigenomes_elaborated_np_enhancers.csv', delimiter=',')
            # promoters = np.loadtxt('epigenomes_elaborated_np_promoters.csv', delimiter=',')
            #
            # enhancers_labels = np.loadtxt('labels_elaborated_enhancers.csv', delimiter=',')
            # promoters_labels = np.loadtxt('labels_elaborated_promoters.csv', delimiter='')

            elaborated_promoters = pd.read_csv('csv/' + cell_line + '/elaborated_promoters.csv', sep=',')
            elaborated_enhancers = pd.read_csv('csv/' + cell_line + '/elaborated_enhancers.csv', sep=',')

            elaborated_promoters_labels = pd.read_csv('csv/' + cell_line + '/labels_elaborated_promoters.csv', sep=',')
            elaborated_enhancers_labels = pd.read_csv('csv/' + cell_line + '/labels_elaborated_enhancers.csv', sep=',')

            epigenomes["promoters"] = elaborated_promoters
            epigenomes["enhancers"] = elaborated_enhancers

            labels["promoters"] = elaborated_promoters_labels
            labels["enhancers"] = elaborated_enhancers_labels

        else:
            epigenomes = dataElaboration(epigenomes, labels, cell_line)

        # Step 3. Data Visualization
        logging.info('Step 3. Data Visualization')
        if are_data_already_visualized(cell_line):
            logging.info('Data already visualized. Skipping Data Visualization.')
        else:
            # TODO check this error ValueError: could not convert string to float: 'chr10'
            # TODO delete chr column (all string column) from dataset because PCA works only with float!
            data_visualization(epigenomes, labels, sequences, cell_line)

        # TODO
        # Step 4. Holdout

        # Step 5. Fit models (Prediction of Active Enhancers/Promoters)

        # Step 6. Measuring Performance

        # Step 7. Statistical Test (Wilcoxon)

        logging.info('Exiting cell_line' + cell_line)

if __name__ == '__main__':
    main()