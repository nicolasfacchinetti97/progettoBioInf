from progettobioinf.entry_point import *


def test_wilcoxon():
    if os.path.exists('results2.json'):
        df = pd.read_json("results2.json")
        run_wilcoxon(df, "FFNN", "CNN", "test")
        run_wilcoxon(df, "FFNN", "MLP", "test")
        run_wilcoxon(df, "MLP", "CNN", "test")