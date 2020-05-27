from epigenomic_dataset import load_epigenomes

# Epigenomic DataSet

cell_line = "K562"
assembly = "hg19"
window_size = 200

promoters_epigenomes, promoters_labels = load_epigenomes(
    cell_line=cell_line,
    dataset="fantom",
    regions="promoters",
    window_size=window_size
)

enhancers_epigenomes, enhancers_labels = load_epigenomes(
    cell_line=cell_line,
    dataset="fantom",
    regions="enhancers",
    window_size=window_size
)

epigenomes = {
    "promoters": promoters_epigenomes,
    "enhancers": enhancers_epigenomes
}

labels = {
    "promoters": promoters_labels,
    "enhancers": enhancers_labels
}