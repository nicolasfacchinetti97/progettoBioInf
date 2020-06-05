from progettobioinf.data_elaboration import *
import logging
logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

def prepare_data(epigenomes, labels, sequences,):
    logging.info("Preparing data")
    tasks = {
        "x": [
            *[
                val
                for val in epigenomes
            ],
            *[
                val.values
                for val in sequences.values()
            ],
            pd.concat(sequences.values()).values,
            pd.concat(sequences.values()).values,
            *[
                np.hstack([
                    pca(epigenomes[region], n_components=25),
                    mfa(sequences[region], n_components=25)
                ])
                for region in epigenomes
            ]
        ],
        "y": [
            *[
                val.values.ravel()
                for val in labels.values()
            ],
            *[
                val.values.ravel()
                for val in labels.values()
            ],
            pd.concat(labels.values()).values.ravel(),
            np.vstack([np.ones_like(labels["promoters"]), np.zeros_like(labels["enhancers"])]).ravel(),
            *[
                val.values.ravel()
                for val in labels.values()
            ],
        ],
        "titles": [
            "Epigenomes promoters",
            "Epigenomes enhancers",
            "Sequences promoters",
            "Sequences enhancers",
            "Sequences active regions",
            "Sequences regions types",
            "Combined promoters data",
            "Combined enhancers data"
        ]
    }

    xs = tasks["x"]
    ys = tasks["y"]
    titles = tasks["titles"]

    assert len(xs) == len(ys) == len(titles)

    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])
    logging.info("Data prepared")
    return xs, ys, titles, colors

def visualization_PCA(xs, ys, titles, colors):
    logging.info("Data visualization PCA")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(32, 16))

    for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
        axis.scatter(*pca(x).T, s=1, color=colors[y])
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        axis.set_title(f"PCA decomposition - {title}")
    plt.savefig('img/pca_decomposition.png')
    logging.info("PCA img saved")


def visualization_TSNE(xs, ys, titles, colors):
    logging.info("Data visualization TSNE")
    for perpexity in tqdm((30, 40, 50, 100, 500, 5000), desc="Running perplexities"):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing TSNEs", total=len(xs)):
            axis.scatter(*cannylab_tsne(x, perplexity=perpexity).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"TSNE decomposition - {title}")
        fig.tight_layout()
        plt.savefig('img/TSNE_decomposition_'+perpexity+'.png')
        logging.info("PCA img saved")