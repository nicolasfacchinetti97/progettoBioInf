import os
import csv
import logging
logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

def initial_setup(cell_line):
    create_img_folder()
    create_img_cell_line_folder(cell_line)
    create_state_folder()
    create_state_cell_line_folder(cell_line)
    create_state_file(cell_line)

def create_img_folder():
    if not os.path.exists('img'):
        os.mkdir('img')
        logging.info('Img folder created')
    else:
        logging.info('Img folder already exists')

def create_img_cell_line_folder(cell_line):
    if not os.path.exists('img/' + cell_line):
        os.mkdir('img/' + cell_line)
        logging.info(cell_line + ' folder created')
    else:
        logging.info(cell_line + ' folder already exists')

def create_state_folder():
    if not os.path.exists('state'):
        os.mkdir('state')
        logging.info('State folder created')
    else:
        logging.info('State folder already exists')

def create_state_cell_line_folder(cell_line):
    if not os.path.exists('state/' + cell_line):
        os.mkdir('state/' + cell_line)
        logging.info('state ' + cell_line + ' folder created')
    else:
        logging.info('state ' + cell_line + ' folder already exists')

def create_state_file(cell_line):
    file_exists = os.path.exists('state/' + cell_line + '/state.txt')
    if file_exists:
        logging.info('State file already exists')
    else:
        filepath = os.path.join('state/' + cell_line + '/', 'state.txt')
        state_file = open(filepath, 'w')
        state_file.write('Step 0. Start processing data')
        state_file.close()
        logging.info('State file created')

def is_state_file_contains_step(step, cell_line):
    with open('state/' + cell_line + '/state.txt') as state_file:
        if step in state_file:
            return True
        else:
            return False

def save_dictionary_as_csv(filename, dictionary):
    w = csv.writer(open(filename,"w"))
    for key, val in dictionary.items():
        w.writerow([key, val])