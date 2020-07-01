from progettobioinf.results import *
import pandas as pd


def test_wilcoxon():
    if os.path.exists('results_tabular_enhancers.json'):
        df = pd.read_json("results_tabular_enhancers.json")
        run_wilcoxon(df, "RandomForestClassifier", "MLP2", "test")