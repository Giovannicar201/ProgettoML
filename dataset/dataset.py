import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#Trasformazione del dataset da dataframe a CSV con aggiunta del nome delle colonne
column=['id','diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1','symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2','symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3','symmetry3','fractal_dimension3']

data_file_path = 'wdbc.data'
dataframe = pd.read_csv(data_file_path, names=column, delimiter=',')

# Salva il dataframe in formato CSV
csv_file_path = 'wdbc.csv'
dataframe.to_csv(csv_file_path, index=False)

dataset=pd.read_csv('wdbc.csv')

print("Lunghezza di dataset", {len(dataset)-1})

for index,column in enumerate(dataset.columns,start=1):
    print(f"{index}. {column}")

#Analisi della classe da seguire
print("Classe per la predizione:")
print(dataset['diagnosis'].unique())

#Ricerca di eventuali dati mancanti
print('Valori mancanti:')
boolean_dataset= dataset.isna()
missing_values = boolean_dataset.sum()
print(missing_values)#Il dataset non presenta valori mancanti per ogni colonna

# Creazione di due dataset, uno per i tumori benigni e uno per quelli maligni
dataset_malignant = dataset[dataset['diagnosis'] == 'M']
dataset_benign = dataset[dataset['diagnosis'] == 'B']

column_to_plot=['diagnosis']

for column in column_to_plot:
    plt.close("all")
    unique_values=dataset[column].unique()
    bins=np.arange(len(unique_values)+1)-0.5

    plt.hist([dataset_benign[column], dataset_malignant[column]], color=["skyblue", "blue"], edgecolor='black', alpha=0.8, stacked=True, rwidth=0.5, bins=bins)

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.legend(["Benign","Malignant"])

    # Salva l'istogramma come immagine
    plt.savefig(f"{column}_histogram.png")

    # Chiudi la figura per liberare la memoria
    plt.close()


column_to_interest = ['radius1', 'texture1', 'perimeter1', 'area1', 'radius2', 'texture2', 'perimeter2', 'area2', 'radius3', 'texture3', 'perimeter3', 'area3']

for column in column_to_interest:
    # Chiusura di tutte le figure "aperte" per evitare sovrapposizioni e visualizzazione distorta degli istogrammi
    plt.close("all")

    # Inizializzazione di un array contenente tutti i valori che la colonna può assumere
    unique_values = dataset[column].unique()
    # Inizializzazione di un array contenente valori da 0 al numero di valori della colonna, che corrisponderà al numero
    # dei bins (barre verticali) contenuti nell'istogramma. Si sottrae 0.5 per centrare il bin rispetto all'etichetta
    bins = np.arange(len(unique_values) + 1) - 0.5

    # Creazione dell'istogramma
    plt.hist([dataset_benign[column], dataset_malignant[column]], color=["skyblue", "blue"], edgecolor='black', alpha=0.8, stacked=True, rwidth=0.5, bins=30)

    # Definizione delle etichette presenti sull'asse x e sull'asse y dell'istogramma
    plt.xlabel(column.replace('-', ' ').capitalize())
    plt.ylabel("Frequenza")
    # Definizione della legenda dell'istogramma
    plt.legend(["Benign","Malignant"])

    # Salva l'istogramma come immagine
    plt.savefig(f"{column}_histogram.png")

    # Chiudi la figura per liberare la memoria
    plt.close()

# Copia il dataset per poter visualizzare i cambiamenti dopo l'encoding
dataset_encoded = dataset.copy()

# Inizializza una lista per memorizzare gli indici dei valori nulli
nan_indices = []

# Label encoding per le colonne categoriche
for column in dataset_encoded.columns:
    if dataset_encoded[column].dtype == 'object':  # Verifica se la colonna è di tipo object (categorica)
        # Salva gli indici dei valori nulli
        nan_indices = dataset_encoded[column][dataset_encoded[column].isna()].index.tolist()

        # Esegui il Label Encoding
        dataset_encoded[column] = LabelEncoder().fit_transform(dataset_encoded[column])

# Ripristino dei valori nulli e conversione in tipo Int64
for column in dataset_encoded.columns:
    if dataset[column].dtype == 'object':  # Verifica se la colonna è di tipo object (categorica)
        dataset_encoded.loc[nan_indices, column] = np.nan
        dataset_encoded[column] = dataset_encoded[column].astype("Int64")

# Visualizza il dataset prima e dopo l'encoding per comprendere le modifiche
print("Dataset prima dell'encoding:")
print(dataset.head())

print("\nDataset dopo l'encoding:")
print(dataset_encoded.head())

#Calcolo delle correlazione del dataset
# Trasforma la colonna 'diagnosis' in valori numerici (0 per 'B' e 1 per 'M')
dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M': 1})

# Calcola la matrice di correlazione
correlation_matrix = dataset.corr()

# Crea la heatmap
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap della correlazione tra le variabili')
plt.savefig(f"heatmap.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
# Posizionamento del pie plot all'interno dell'immagine
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
# Creazione del pie plot per controllare il bilanciamento della classe da predire
plt.pie(dataset['diagnosis'].value_counts(), labels=['Benign', 'Malignant'], colors=["skyblue", "blue"],explode=(0, 0.015), autopct="%0.2f", startangle=90, textprops={'fontsize': 11})
plt.savefig(f"pieplot.png")
plt.close()