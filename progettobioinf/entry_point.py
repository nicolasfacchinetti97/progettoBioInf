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
        if(is_state_file_contains_step('elaboration_step', cell_line)):
            logging.info('Skipping Data Elaboration')
        else:
            epigenomes = dataElaboration(epigenomes, labels)

        # Step 3. Data Visualization
        logging.info('Step 3. Data Visualization')
        if(is_state_file_contains_step('visualization_step', cell_line)):
            logging.info('Skipping Data Visualization')
        else:
            data_visualization(epigenomes, labels, sequences)

        # Step 4. Holdout

        # Step 5. Fit models (Prediction of Active Enhancers/Promoters)

        # Step 6. Measuring Performance

        # Step 7. Statistical Test (Wilcoxon)

        logging.info('Exiting cell_line' + cell_line)

if __name__ == '__main__':
    main()