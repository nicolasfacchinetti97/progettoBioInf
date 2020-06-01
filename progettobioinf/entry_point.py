from progettobioinf.initial_setup import *
from progettobioinf.data_processing import *

if __name__ == '__main__':

    cell_lines = ["K562"]
    assembly = "hg19"
    window_size = 200

    for cell_line in cell_lines:
        print('Cell Line: ' + cell_line)

        # Step 0. Initial Setup
        print('Step 0. Initial Setup')
        initial_data = initial_setup(cell_line)

        # Step 1. Data Retrieval
        print('Step 1. Data Retrieval')
        data = dataRetrieval(cell_line, assembly, window_size)
        epigenomes = data[0]
        labels = data[1]
        sequences = data[2]

        # Step 2. Data Elaboration
        print('Step 2. Data elaboration')
        elaborated_data = dataElaboration(epigenomes, labels, cell_line)

        # Step 3. Data Visualization
        print('Step 3. Data Visualization')
        data_visualization(epigenomes, labels, sequences)
