import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def train_and_evaluate(model, model_intial,X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_true=y_train, y_pred=train_pred)
    precision_train = precision_score(y_true=y_train, y_pred=train_pred,pos_label='M')
    recall_train = recall_score(y_true=y_train, y_pred=train_pred,pos_label='M')
    f1_train = f1_score(y_true=y_train, y_pred=train_pred,pos_label='M')

    accuracy_test = accuracy_score(y_true=y_test, y_pred=test_pred)
    precision_test = precision_score(y_true=y_test, y_pred=test_pred,pos_label='M')
    recall_test = recall_score(y_true=y_test, y_pred=test_pred,pos_label='M')
    f1_test = f1_score(y_true=y_test, y_pred=test_pred,pos_label='M')

    print(f"Training\n"
          f"Accuracy: {round(accuracy_train,2)}\n"
          f"Precision: {round(precision_train,2)}\n"
          f"recall: {round(recall_train,2)}\n"
          f"F1: {round(f1_train,2)}")

    print(f"Testing\n"
          f"Accuracy: {round(accuracy_test,2)}\n"
          f"Precision: {round(precision_test,2)}\n"
          f"recall: {round(recall_test,2)}\n"
          f"F1: {round(f1_test,2)}")

    save_evaluation_graphs(y_test, test_pred, model_intial)

def save_evaluation_graphs(real_values, pred, model_initial):
    plt.close("all")
    # Creazione della confusion matrix relativa al modello Naive Bayes
    matrix = confusion_matrix(y_true=real_values, y_pred=pred)

    color_map = ListedColormap('white', name='colormap_list')
    color_matrix = [['skyblue', 'blue'], ['blue','skyblue']]
    color_text_matrix = [['black', 'white'], ['white', 'black']]

    # Visualizzazione della confusion matrix
    plt.imshow(matrix, cmap=color_map, origin='upper')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Definizione dei dettagli in merito a colore e testo per le celle della matrice
            plt.text(j, i, str(matrix[i, j]), color=color_text_matrix[i][j])
            plt.fill_between([j - 0.5, j + 0.5], i - 0.5, i + 0.5, color=color_matrix[i][j], alpha=1)

    # Definizione dei valori e delle etichette presenti sull'asse x e sull'asse y della confusion matrix
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.savefig(f"{model_initial}_confusion_matrix.png")

    plt.close("all")
    # Codifica dei valori di stringa in numeri
    y_test_numeric = real_values.replace({'M': 1, 'B': 0})
    pred_numeric = pd.Series(pred).replace({'M': 1, 'B': 0})
    # Calcolo del tasso di falsi positivi, veri positivi e le soglie
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test_numeric, pred_numeric)
    # Disegno della linea di riferimento
    plt.plot([0, 1], [0, 1], 'k--')
    # Disegno della ROC Curve, etichettandola con il valore della AUC (più vicino è a 1, migliore è il modello)
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.4f})'.format(auc(false_positive_rate, true_positive_rate)), color="#D50630")
    # Definizione delle etichette presenti sull'asse x e sull'asse y della ROC Curve
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')

    plt.savefig(f"{model_initial}_roc_curve.png")

dataset_path=f"C:/Users/Darkn/Desktop/Progetto ML/pythonProject/dataset/wdbc.csv"
dataset = pd.read_csv(dataset_path)

# Ottenimento delle labels
y = dataset["diagnosis"]
# Ottenimento del dataset senza labels
X = dataset.drop("diagnosis", axis=1)

# Data splitting casuale per ottenere il dataset di training e il dataset di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Inizializzazione dei modelli
nb = MultinomialNB()
lr = LogisticRegression(max_iter=200)

# Addestramento dei modelli e valutazione delle performance
#Naive Bayes
train_and_evaluate(nb, 'Naive Bayes',X_train, X_test, y_train, y_test)
#Logistic Regression
train_and_evaluate(lr, 'Logistic Regression',X_train, X_test, y_train, y_test)