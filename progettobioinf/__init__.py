from download_data import get_promoters_enhancers, get_sequence_data

cell_line = "K562"                      # MCF-7
assembly = "hg19"                       # human genome
window_size = 200                       # the width of the window of nucleotides we are going to take in consideration


if __name__ == "__main__":
    # get epigenetic data
    epigenomes, labels = get_promoters_enhancers(cell_line, window_size)
    # get sequences data
    sequences = get_sequence_data(assembly, window_size, epigenomes)

    print(sequences["promoters"][:2])
