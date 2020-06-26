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
        height=5,
        show_legend=False,
        path=path
    )


def run_wilcoxon(df, model_a, model_b, run_type):
    logging.info("Check statistical test - Wilcoxon [{}, {}] {}".format(model_a, model_b, run_type))
    f = open("log/info.txt", "a+")
    f.write("Run type {} - Wilcoxon [{}, {}] \n".format(run_type.upper() , model_a, model_b))
    f.close()
    models = df[
        (df.run_type == run_type)
    ]

    ffnn_scores = models[models.model == model_a]
    mlp_scores = models[models.model == model_b]

    alpha = 0.01

    for metric in ffnn_scores.columns[-4:]:
        if metric == "Accuracy" or metric == "Balanced Accuracy" or metric == "accuracy" or metric == "loss":
            continue
        logging.info(metric)
        a, b = ffnn_scores[metric], mlp_scores[metric]

        stats, p_value = wilcoxon(a, b)
        if p_value > alpha:
            logging.info("The two models performance are statistically identical: {}".format(p_value))
            f = open("log/info.txt", "a+")
            f.write("{}, The two models performance are statistically identical: {} \n".format(metric, p_value))
            f.close()
        else:
            logging.info("The two models performance are different: {}".format(p_value))
            f = open("log/info.txt", "a+")
            f.write("{}, The two models performance are different: {} \n".format(metric, p_value))
            if a.mean() > b.mean():
                logging.info("The first model is better")
                f.write("The first model is better" + "\n")
            else:
                logging.info("The second model is better")
                f.write("The second model is better" + "\n")
            f.close()
    f = open("log/info.txt", "a+")
    f.write("{}\n".format("="*50))
    f.close
