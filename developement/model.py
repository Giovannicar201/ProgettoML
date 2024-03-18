import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap

def train_and_evaluate(model, model_initial, X_train, X_test, y_train, y_test):
    # Addestramento del modello
    model.fit(X_train, y_train)

    # Predizione sul training set e sul test set
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calcolo delle metriche di valutazione prima dell'oversampling
    accuracy_train_before = accuracy_score(y_true=y_train, y_pred=train_pred)
    precision_train_before = precision_score(y_true=y_train, y_pred=train_pred, pos_label='M')
    recall_train_before = recall_score(y_true=y_train, y_pred=train_pred, pos_label='M')
    f1_train_before = f1_score(y_true=y_train, y_pred=train_pred, pos_label='M')

    accuracy_test_before = accuracy_score(y_true=y_test, y_pred=test_pred)
    precision_test_before = precision_score(y_true=y_test, y_pred=test_pred, pos_label='M')
    recall_test_before = recall_score(y_true=y_test, y_pred=test_pred, pos_label='M')
    f1_test_before = f1_score(y_true=y_test, y_pred=test_pred, pos_label='M')

    # Applica l'oversampling con SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Calcola la distribuzione delle classi prima e dopo l'oversampling
    class_distribution_before = y_train.value_counts(normalize=True)
    class_distribution_after = y_train_resampled.value_counts(normalize=True)

    # Visualizza graficamente la distribuzione delle classi
    plt.figure(figsize=(10, 5))
    plt.bar(class_distribution_before.index, class_distribution_before, color='skyblue', label='Before SMOTE')
    plt.bar(class_distribution_after.index, class_distribution_after, color='blue', label='After SMOTE')
    plt.title('Class Distribution Before and After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Proportion')
    plt.xticks(class_distribution_before.index, ['Benign', 'Malignant'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_initial}_class_distribution.png")
    plt.close()

    # Creazione del pie plot per la distribuzione delle classi
    plt.figure(figsize=(6, 6))
    labels = ['Benign', 'Malignant']
    sizes = [y_train.value_counts()[0], y_train.value_counts()[1]]
    colors = ['skyblue', 'blue']
    explode = (0, 0.1)  # Esplosione della fetta "Malignant"
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%0.2f", startangle=90, textprops={'fontsize': 11})
    plt.axis('equal')  # Equalizza gli assi per ottenere un cerchio
    plt.title('Class Distribution Before SMOTE')
    plt.savefig(f"{model_initial}_class_distribution_pie.png")
    plt.close()

    # Creazione del pie plot per la distribuzione delle classi dopo l'oversampling
    plt.figure(figsize=(6, 6))
    labels = ['Benign', 'Malignant']
    sizes = [y_train_resampled.value_counts()[0], y_train_resampled.value_counts()[1]]
    colors = ['skyblue', 'blue']
    explode = (0, 0.1)  # Esplosione della fetta "Malignant"
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,  autopct="%0.2f", startangle=90, textprops={'fontsize': 11})
    plt.axis('equal')  # Equalizza gli assi per ottenere un cerchio
    plt.title('Class Distribution After SMOTE')
    plt.savefig(f"{model_initial}_class_distribution_pie_after_smote.png")
    plt.close()

    # Addestramento del modello sui dati oversampled
    model.fit(X_train_resampled, y_train_resampled)

    # Predizione sui dati oversampled
    train_pred_resampled = model.predict(X_train_resampled)
    test_pred_resampled = model.predict(X_test)

    # Calcolo delle metriche di valutazione dopo l'oversampling
    accuracy_train_after = accuracy_score(y_true=y_train_resampled, y_pred=train_pred_resampled)
    precision_train_after = precision_score(y_true=y_train_resampled, y_pred=train_pred_resampled, pos_label='M')
    recall_train_after = recall_score(y_true=y_train_resampled, y_pred=train_pred_resampled, pos_label='M')
    f1_train_after = f1_score(y_true=y_train_resampled, y_pred=train_pred_resampled, pos_label='M')

    accuracy_test_after = accuracy_score(y_true=y_test, y_pred=test_pred_resampled)
    precision_test_after = precision_score(y_true=y_test, y_pred=test_pred_resampled, pos_label='M')
    recall_test_after = recall_score(y_true=y_test, y_pred=test_pred_resampled, pos_label='M')
    f1_test_after = f1_score(y_true=y_test, y_pred=test_pred_resampled, pos_label='M')

    # Stampa delle metriche di valutazione prima e dopo l'oversampling
    print("Metrics before oversampling:")
    print(f"Training\n"
          f"Accuracy: {round(accuracy_train_before,2)}\n"
          f"Precision: {round(precision_train_before,2)}\n"
          f"Recall: {round(recall_train_before,2)}\n"
          f"F1: {round(f1_train_before,2)}")
    print(f"Testing\n"
          f"Accuracy: {round(accuracy_test_before,2)}\n"
          f"Precision: {round(precision_test_before,2)}\n"
          f"Recall: {round(recall_test_before,2)}\n"
          f"F1: {round(f1_test_before,2)}")

    print("\nMetrics after oversampling:")
    print(f"Training\n"
          f"Accuracy: {round(accuracy_train_after,2)}\n"
          f"Precision: {round(precision_train_after,2)}\n"
          f"Recall: {round(recall_train_after,2)}\n"
          f"F1: {round(f1_train_after,2)}")
    print(f"Testing\n"
          f"Accuracy: {round(accuracy_test_after,2)}\n"
          f"Precision: {round(precision_test_after,2)}\n"
          f"Recall: {round(recall_test_after,2)}\n"
          f"F1: {round(f1_test_after,2)}")

    # Mostra la confusion matrix e la ROC curve dopo l'oversampling
    save_evaluation_graphs(y_test, test_pred_resampled, model_initial)

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