import logging
import os
from glob import glob

import pandas as pd
from PIL import Image
from barplots import barplots
from scipy.stats import wilcoxon

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


def get_wilcoxon(df):
    # Here we will be doing a statistical test.
    models = df[
        (df.run_type == "test")
    ]

    ffnn_scores = models[models.model == "FFNN"]
    mlp_scores = models[models.model == "MLP"]

    alpha = 0.01

    for metric in ffnn_scores.columns[-4:]:
        logging.info(metric)
        a, b = ffnn_scores[metric], mlp_scores[metric]
        stats, p_value = wilcoxon(a, b)
        if p_value > alpha:
            logging.info("The two models performance are statistically identical: {}".format(p_value))
        else:
            logging.info("The two models performance are different: {}".format(p_value))
            if a.mean() > b.mean():
                logging.info("The first model is better")
            else:
                logging.info("The second model is better")

# TODO
# def check_statistical_tests():
#     logging.info("check statistical test")
