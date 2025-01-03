import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def calcular_matriz_confusion_usr(test, user_item_matriu, threshold=4.0):
    """
    Calcula i mostra la matriu de confusió per al recomanador user-user després de convertir a binari.

    Args:
        test (DataFrame): Dades de prova amb "User ID", "Product ID" i "Rating".
        user_item_matriu (DataFrame): Matriu usuari-producte amb prediccions.
        threshold (float): Llindar per considerar una valoració com a positiva.

    Returns:
        dict: Diccionari amb FP, TP, FN, TN.
    """
    # Filtrar dades vàlides
    dades_valides = test[
        test['User ID'].isin(user_item_matriu.index) &
        test['Product ID'].isin(user_item_matriu.columns)
    ]

    if dades_valides.empty:
        raise ValueError("El conjunt de dades de prova no té valors comuns amb la matriu usuari-producte.")

    # Obtenir les prediccions fent servir una indexació eficient
    prediccions = dades_valides.apply(
        lambda fila: user_item_matriu.at[fila['User ID'], fila['Product ID']], axis=1
    )

    # Convertir valoracions reals i prediccions a binari
    valoracions_reals = (dades_valides['Rating'] >= threshold).astype(int).values
    prediccions_binàries = (prediccions >= threshold).astype(int).values

    # Calcular la matriu de confusió
    tn, fp, fn, tp = confusion_matrix(valoracions_reals, prediccions_binàries).ravel()

    # Mostrar resultats
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)
    print("True Negatives (TN):", tn)

    # Mostrar la matriu de confusió en format visual
    matriz = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots()
    cax = ax.matshow(matriz, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    
    labels = [["TN", "FP"], ["FN", "TP"]]
    for (i, j), val in np.ndenumerate(matriz):
        ax.text(j, i, f'{labels[i][j]}', ha='center', va='center', fontsize=12, weight='bold')
        ax.text(j, i + 0.3, f'({val})', ha='center', va='center', fontsize=10, color='gray')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Negative (0)', 'Positive (1)'])
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title("Confusion Matrix for User-User Recommender")
    plt.show(block=False)  # Mostra el gràfic sense bloquejar l'execució
    input("Prem Enter per continuar...")  # Manté la finestra oberta fins que es prem Enter


    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def calcular_accuracy(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    return (tp + tn) / total

def calcular_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calcular_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calcular_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0




ruta = r'C:\Users\usuari\Desktop\3r dades\APRENENTATGE COMPUTACIONAL\PROJECTE ACVELL\Reviews.csv'
data = pd.read_csv(ruta)

data = data.rename(columns={
    'UserId': 'User ID',
    'ProductId': 'Product ID',
    'Score': 'Rating'
})

total_ratings_1 = data.shape[0]
total_usuaris_1 = data['User ID'].nunique()
total_productes_1 = data['Product ID'].nunique()

rating_counts_before = data['Rating'].value_counts().sort_index()

data = data.dropna()  
data = data[(data['Rating'] >= 1) & (data['Rating'] <= 5)]  
usuaris_amb_mes_de_10_reviews = data.groupby('User ID').size()
usuaris_adequats = usuaris_amb_mes_de_10_reviews[usuaris_amb_mes_de_10_reviews >= 10].index

# Filtrar les valoracions només per aquests usuaris
data_filtrada = data[data['User ID'].isin(usuaris_adequats)]

# Filtratge dels productes amb almenys 2 valoracions
productes_amb_mes_de_2_reviews = data_filtrada.groupby('Product ID').size()
productes_adequats = productes_amb_mes_de_2_reviews[productes_amb_mes_de_2_reviews >= 2].index

# Filtrar les dades per incloure només els productes amb almenys 2 valoracions
data_retallat = data_filtrada[data_filtrada['Product ID'].isin(productes_adequats)]


# Distribució de les valoracions després de netejar el dataset
rating_counts_after = data_filtrada['Rating'].value_counts().sort_index()

data_retallat = data_retallat[['User ID', 'Product ID', 'Rating']]



# Dividir el conjunt en train (60%), test (20%) i validació (20%)
train, temp = train_test_split(data_retallat, test_size=0.4, random_state=42)  # 60% train, 40% temporal
test, validation = train_test_split(temp, test_size=0.5, random_state=42)  # 50% del temporal per a test i validació


# USER USER
train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
user_item_matriu = train.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)
test = test[(test['User ID'].isin(user_item_matriu.index)) & (test['Product ID'].isin(user_item_matriu.columns))]

# Calcular i mostrar la matriu de confusió
resultats = calcular_matriz_confusion_usr(test, user_item_matriu, threshold=3.0)
accuracy = calcular_accuracy(resultats["TP"], resultats["FP"], resultats["FN"], resultats["TN"])
precision = calcular_precision(resultats["TP"], resultats["FP"])
recall = calcular_recall(resultats["TP"], resultats["FN"])
f1_score = calcular_f1_score(precision, recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
