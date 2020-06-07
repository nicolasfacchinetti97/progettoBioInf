import logging
import os
from glob import glob

import pandas as pd
from PIL import Image
from barplots import barplots

logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


def save_results_df_to_csv(df_results, cell_line):
    if os.path.exists('csv/' + cell_line + '/results.csv'):
        logging.info('results already exists')
    else:
        df_results.to_csv('csv/' + cell_line + '/results.csv', sep=',')


def convert_results_to_dataframe(results):
    df_results = pd.DataFrame(results)
    df_results = df_results.drop(columns=["holdout"])
    return df_results


def setup_barplot(df_results):
    barplots(
        df_results,
        groupby=["model", "run_type"],
        show_legend=False,
        height=5,
        orientation="horizontal"
    )


def save_barplots_to_png():
    setup_barplot()

    for x in glob("barplots/*.png"):
        im = Image.open(x)
        im.save("img/" + x)  # TODO controlla questo

# TODO
# def check_statistical_tests():
#     logging.info("check statistical test")
