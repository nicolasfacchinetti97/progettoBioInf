import os

# Create img folder if not exists
def initial_setup(cell_line):
    create_img_folder()
    create_img_cell_line_folder(cell_line)

def create_img_folder():
    if not os.path.exists('img'):
        os.mkdir('img')
        print('Img folder created')
    else:
        print('Img folder already exists')

def create_img_cell_line_folder(cell_line):
    if not os.path.exists('img/' + cell_line):
        os.mkdir('img/' + cell_line)
        print(cell_line + ' folder created')
    else:
        print(cell_line + ' folder already exists')
