from multiprocessing import cpu_count

from boruta import BorutaPy
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier

from first_elaboration import *
from plot_data_image import *
from remove_uncorrellated_data import *
from initial_setup import *




def elaborate_epigenomics_data(epigenomes, labels, cell_line):
    ## Rate between features and samples, check if the datasets have a rate of features and samples greater than one
    logging.info("Rate features samples")
    rate_features_samples(epigenomes)

    ## NaN detection, drop feature if more than 10% of NaN values
    logging.info("NaN Detection")
    nan_detection(epigenomes)

    ## KNN imputation, remova NaN values
    logging.info("KNN imputation")
    epigenomes = knn_imputation(epigenomes)

    ## Class Balance
    logging.info("Checking class balance")
    check_class_balance(labels, cell_line)

    ## Drop constant features, these do not add any additional value therefore drop it.
    logging.info("Dropping constant features")
    epigenomes = drop_constant_features(epigenomes)

    ## Z-Scoring (robust scaler)
    logging.info("Data normalization")
    epigenomes = data_normalization(epigenomes)

    # ====================== CORRELATION WITH OUPUT ======================
    p_value_threshold = 0.01
    correlation_threshold = 0.05

    ## Pearson (LINEAR)
    logging.info("Executing Pearson Test")
    uncorrelated_pearson = execute_pearson(epigenomes, labels, p_value_threshold)
    epigenomes = drop_features(epigenomes, uncorrelated_pearson)

    ## Spearman (MONOTONIC)
    logging.info("Executing Spearman Test")
    uncorrelated_spearman = execute_spearman(epigenomes, labels, p_value_threshold)
    epigenomes = drop_features(epigenomes, uncorrelated_spearman)

    ## MIC (NON-LINEAR)
    logging.info("Executing Mic Test")
    uncorrelated_mic = execute_mic(epigenomes, labels, correlation_threshold)
    epigenomes = drop_features(epigenomes, uncorrelated_mic)

    for region, x in epigenomes.items():
        logging.info('Saving epigenomes elaborated csv (' + region + ')')
        epigenomes[region].to_csv('csv/' + cell_line + '/elaborated_' + region + '.csv', sep=',')

    return epigenomes


def do_features_correlations(epigenomes, labels, cell_line, number_tuples, top_number):
    # ====================== Features correlations ======================
    logging.info("Checking features correlations")

    p_value_threshold = 0.01
    correlation_threshold = 0.95
    # TODO COSA FARE CON LE TUPLE CHE TORNA? VANNO TOLTE???? QUESTA FUNZIONE VA MESSA SOPRA????
    extremely_correlated, scores = check_features_correlations(epigenomes, p_value_threshold, correlation_threshold)

    # Sort the obtained scores
    logging.info("Sorting the obtained scores")
    scores = {
        region: sorted(score, key=lambda x: np.abs(x[0]), reverse=True)
        for region, score in scores.items()
    }

    ## Scatter plot (most correlated touples)
    logging.info("Scatter plot - detect most n correlated touples")
    detect_most_n_correlated_touples(epigenomes, scores, number_tuples, labels, cell_line)

    ## Scatter plot (most uncorrelated touples)
    logging.info("Scatter plot - detect most n uncorrelated touples")
    detect_most_n_uncorrelated_touples(epigenomes, scores, number_tuples, labels, cell_line)

    # Features distributions
    logging.info("Getting top n different features")
    get_top_n_different_features(epigenomes, labels, top_number, cell_line)

    logging.info("Getting top n different touples")
    get_top_n_different_tuples(epigenomes, top_number, cell_line)


# Features correlations
def check_features_correlations(epigenomes, p_value_threshold, correlation_threshold):
    extremely_correlated = {
        region: set()
        for region in epigenomes
    }

    scores = {
        region: []
        for region in epigenomes
    }
    for region, x in epigenomes.items():
        for i, column in tqdm(
                enumerate(x.columns),
                total=len(x.columns), desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            for feature in x.columns[i + 1:]:
                correlation, p_value = pearsonr(x[column].values.ravel(), x[feature].values.ravel())
                correlation = np.abs(correlation)
                scores[region].append((correlation, column, feature))
                if p_value < p_value_threshold and correlation > correlation_threshold:
                    if entropy(x[column]) > entropy(x[feature]):
                        extremely_correlated[region].add(feature)
                    else:
                        extremely_correlated[region].add(column)
    return extremely_correlated, scores


# Feature Selection with Boruta
def get_features_filter(X: pd.DataFrame, y: pd.DataFrame) -> BorutaPy:
    boruta_selector = BorutaPy(
        RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5),
        n_estimators='auto',
        verbose=2,
        alpha=0.05,  # p_value
        max_iter=10,  # In practice one would run at least 100-200 times
        random_state=42
    )
    boruta_selector.fit(X.values, y.values.ravel())
    return boruta_selector


def start_feature_selection(epigenomes, labels):
    filtered_epigenomes = {
        region: get_features_filter(
            X=x,
            y=labels[region]
        ).transform(x.values)

        for region, x in tqdm(
            epigenomes.items(),
            desc="Running Boruta Feature estimation"
        )
    }
    return filtered_epigenomes
