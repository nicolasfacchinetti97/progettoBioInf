# Progetto di Bioinformatica
Questo progetto ha come obiettivo la predizione di regioni regolatorie attive (promoters, enhancers).


## Main
E' stato creato l'entry_point.py che è il punto di ingresso del nostro progetto.
Questo contiene il main, al cui interno per ogni linea cellulare vengono eseguiti vari step:
- Step 0. Initial Setup: vengono create alcune cartelle di default
- Step 1. Data Retrieval: vengono recuperati i dati epigenomici (pipeline load_epigenomes) e dati di sequenza (ucsc_genomes_downloader)
- Step 2. Data Elaboration: i dati recuperati vengono elaborati
- Step 3. Data Visualization: visualizzazione dei dati
...

## Data Processing
E' stato creato il "data_processing.py" al cui interno sono dichiarate le funzioni:
- dataRetrieval
- dataElaboration
- dataVisualization
...

Queste funzioni vengono utilizzate dall'entry_point.py.


## Data Retrieval
Tramite epigenomic_dataset (load_epigenomes) e ucsc_genomes_downloader (Genome) vengono recuperati rispettivamente dati epigenomici e dati di sequenza.
Memorizziamo dati epigenomici in un dizionario (epigenomes) le cui chiavi sono "promoters" ed "enhancers", i valori sono i dataframe recuperati (rispettivamente promoters_epigenomes ed enhancers_epigenomes).
Memorizziamo allo stesso modo le labels (dizionario labels con dataframe promoters_labels ed enhancers_labels)
Memorizziamo infine i dati di sequenza all'interno del dizionario sequences.

In seguito, per comodità, andiamo a memorizzare i dataframe ottenuti in csv (df.to_csv).

## Data Elaboration
Se i dati non sono già stati elaborati (in tal caso vengono recuperati direttamente dai csv), si procede con l'elaborazione dati:
- Rate features samples
- NaN Detection
- KNN imputation
- Class balance
- Dropping constant features
- Data Normalization (z-scoring)
- Pearson Test
- Spearman Test
- MIC Test
- Dropping features uncorrelated with output
- Checking features correlations
- Scatter plot (most correlated/uncorrelated touples)
- Scatter plot (get top different features/touples)
- Feature selection (boruta)

In seguito, per comodità, i dati appena elaborati vengono memorizzati all'interno di un csv:
- epigenomes_cell_line_promoters.csv
- epigenomes_cell_line_enhancers.csv

## Data Visualization
E' stato utilizzato il metodo di visualizzazione PCA attraverso il quale andiamo a confrontare dati epigenomici, di sequenza e li andiamo a combinare.

## Logger
E' stato impostato un logger in ogni file che consente di stampare a livello INFO (con data e ora).

## Image
Ogni grafico ottenuto viene memorizzato all'interno della cartella img/cell_line
