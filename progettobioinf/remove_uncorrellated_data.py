# Correlation with output

from minepy import MINE
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from initial_setup import *


## Linear Correlation (Pearson)
def execute_pearson(epigenomes, labels, p_value_threshold):
    uncorrelated = {
        region: set()
        for region in epigenomes
    }
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = pearsonr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                logging.info("Pearson test: remove column {}".format(column))
                f = open("log/info.txt", "a+")
                f.write("Pearson test: remove column " + str(column) + " \n")
                f.close()
                uncorrelated[region].add(column)
    return uncorrelated


## Monotonic Correlation (Spearman)
def execute_spearman(epigenomes, labels, p_value_threshold):
    uncorrelated = {
        region: set()
        for region in epigenomes
    }
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = spearmanr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                logging.info("Sperman test: remove column {}".format(column))
                f = open("log/info.txt", "a+")
                f.write("Spearman test: remove column " + str(column) + " \n")
                f.close()
                uncorrelated[region].add(column)
    return uncorrelated


## Non-Linear Correlation (MIC)
def execute_mic(epigenomes, labels, correlation_threshold):
    uncorrelated = {
        region: set()
        for region in epigenomes
    }
    for region, x in epigenomes.items():
        for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True,
                           leave=False):
            mine = MINE()
            mine.compute_score(x[column].values.ravel(), labels[region].values.ravel())
            score = mine.mic()
            if score >= correlation_threshold:
                logging.info("Non-Linear test: remove column {}".format(column))
                f = open("log/info.txt", "a+")
                f.write("Non-Linear test: remove column " + str(column) + " \n")
                f.close()
                uncorrelated[region].remove(column)
    return uncorrelated


### Drop features uncorrelated with output
def drop_features(epigenomes, uncorrelated):
    for region, x in epigenomes.items():
        epigenomes[region] = x.drop(columns=[
            col
            for col in uncorrelated[region]
            if col in x.columns
        ])
    return epigenomes
