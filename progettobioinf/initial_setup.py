import csv
import logging
import os


def initial_setup(cell_line):
    create_initial_folders(cell_line)


def create_initial_folders(cell_line):
    create_img_folder()
    create_img_cell_line_folder(cell_line)
    create_csv_folder()
    create_csv_cell_line_folder(cell_line)
    create_json_folder()
    create_json_cell_line_folder(cell_line)
    create_log_folder()


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


def create_json_folder():
    if not os.path.exists('json'):
        os.mkdir('json')
        logging.info('json folder created')
    else:
        logging.info('json folder already exists')


def create_json_cell_line_folder(cell_line):
    if not os.path.exists('json/' + cell_line):
        os.mkdir('json/' + cell_line)
        logging.info('json ' + cell_line + ' folder created')
    else:
        logging.info('json ' + cell_line + ' folder already exists')


def create_log_folder():
    if not os.path.exists('log'):
        os.mkdir('log')
        logging.info('log folder created')
    else:
        logging.info('log folder already exists')


def create_log_file():
    if not os.path.exists('/log/info.txt'):
        f = open("info.txt", "w+")
        f.write("Started")
        f.close()
        logging.info("Log file created")
    else:
        logging.info('Log file already exists')


def save_dictionary_as_csv(filename, dictionary):
    w = csv.writer(open(filename, "w"))
    for key, val in dictionary.items():
        w.writerow([key, val])


# setup logging
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
