import os
import csv
import logging

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def initial_setup(cell_line):
    create_initial_folders(cell_line)


def create_initial_folders(cell_line):
    create_img_folder()
    create_img_cell_line_folder(cell_line)
    create_csv_folder()
    create_csv_cell_line_folder(cell_line)


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


def create_csv_folder():
    if not os.path.exists('csv'):
        os.mkdir('csv')
        logging.info('csv folder created')
    else:
        logging.info('csv folder already exists')


def create_csv_cell_line_folder(cell_line):
    if not os.path.exists('csv/' + cell_line):
        os.mkdir('csv/' + cell_line)
        logging.info('csv ' + cell_line + ' folder created')
    else:
        logging.info('csv ' + cell_line + ' folder already exists')


def save_dictionary_as_csv(filename, dictionary):
    w = csv.writer(open(filename, "w"))
    for key, val in dictionary.items():
        w.writerow([key, val])
