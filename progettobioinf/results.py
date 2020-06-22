from barplots import barplots
from scipy.stats import wilcoxon

from initial_setup import *


def save_results_df_to_csv(df_results, cell_line):
    if os.path.exists('csv/' + cell_line + '/results.csv'):
        logging.info('results already exists')
    else:
        df_results.to_csv('csv/' + cell_line + '/results.csv', sep=',')


def generate_barplots(df, path, region):
    logging.info("Generating barplots images for region {}".format(region))
    barplots(
        df,
        groupby=["model", "run_type"],
        orientation="horizontal",
        show_legend=False,
        path=path
    )


def get_wilcoxon(df):
    logging.info("Check statistical test - Wilcoxon")
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
