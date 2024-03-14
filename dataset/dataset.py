import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Trasformazione del dataset da dataframe a CSV con aggiunta del nome delle colonne
column=['id','diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1','symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2','symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3','symmetry3','fractal_dimension3']

data_file_path = 'wdbc.data'
dataframe = pd.read_csv(data_file_path, names=column, delimiter=',')

# Salva il dataframe in formato CSV
csv_file_path = 'wdbc.csv'
dataframe.to_csv(csv_file_path, index=False)

print(f'Trasformazione completata. File CSV salvato in: {csv_file_path}')

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
print(missing_values)
#Il dataset non presenta valori mancanti per ogni colonna

#Creazione di due dataset uno per i tumori beningni e uno per quelli maligni
dataset_malignant=dataset[dataset['diagnosis']=='M']
dataset_benign=dataset[dataset['diagnosis']=='B']
