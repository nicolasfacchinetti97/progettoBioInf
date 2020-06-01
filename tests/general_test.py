from progettobioinf.dataprocessing.initial_setup import *
from tests import *

def test_create_folder():
    create_img_folder()

def test_save_img_plot():
    plt.plot([0, 1, 2, 3, 4], [0, 4, 5, 8, 12])
    plt.savefig('img/test_plot.png')