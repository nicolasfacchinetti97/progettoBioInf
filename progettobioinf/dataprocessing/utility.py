import os

# Create img folder if not exists
def create_img_folder():
    if not os.path.exists('img'):
        os.mkdir('img')
        print('Img folder created')
    else:
        print('Img folder already exists')