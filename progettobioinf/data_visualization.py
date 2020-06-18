from sklearn.decomposition import PCA

from data_elaboration import *





def prepare_data(epigenomes, labels, sequences):
    logging.info("Preparing data")
    tasks = {
        "x": [
            *[
                val.values
                for val in epigenomes.values()
            ],
            *[
                val.values
                for val in sequences.values()
            ],
            pd.concat(sequences.values()).values,
            pd.concat(sequences.values()).values,
            *[
                np.hstack([
                    pca(epigenomes[region].to_numpy(), n_components=25)
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

    return xs, ys, titles, colors


def visualization_PCA(xs, ys, titles, colors, cell_line):
    logging.info("Data visualization PCA")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(32, 16))

    for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
        axis.scatter(*pca(x).T, s=1, color=colors[y])
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        axis.set_title(f"PCA decomposition - {title}")
    plt.savefig('img/' + cell_line + '/pca_decomposition.png')
    logging.info("PCA img saved")


def pca(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(x)

