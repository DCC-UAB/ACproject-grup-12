import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


ruta = r'C:\Users\usuari\Desktop\3r dades\APRENENTATGE COMPUTACIONAL\PROJECTE ACVELL\Reviews.csv'
data = pd.read_csv(ruta)

# Renombrar columnes segons el nou dataset
data = data.rename(columns={
    'UserId': 'User ID',
    'ProductId': 'Product ID',
    'Score': 'Rating'
})


# ESTADÍSTIQUES INICIALS 
total_ratings_1 = data.shape[0]
total_usuaris_1 = data['User ID'].nunique()
total_productes_1 = data['Product ID'].nunique()

rating_counts_before = data['Rating'].value_counts().sort_index()

data = data.dropna()  # Elimina les files amb valors nuls
data = data[(data['Rating'] >= 1) & (data['Rating'] <= 5)]  # Filtra les classificacions invàlides

usuaris_amb_mes_de_10_reviews = data.groupby('User ID').size()

usuaris_adequats = usuaris_amb_mes_de_10_reviews[usuaris_amb_mes_de_10_reviews >= 10].index

data_filtrada = data[data['User ID'].isin(usuaris_adequats)]
productes_amb_mes_de_2_reviews = data_filtrada.groupby('Product ID').size()
productes_adequats = productes_amb_mes_de_2_reviews[productes_amb_mes_de_2_reviews >= 2].index

data_retallat = data_filtrada[data_filtrada['Product ID'].isin(productes_adequats)]

rating_counts_after = data_filtrada['Rating'].value_counts().sort_index()

data_retallat = data_retallat[['User ID', 'Product ID', 'Rating']]

# Dividir el conjunt en train (60%), test (20%) i validació (20%)
train, temp = train_test_split(data_retallat, test_size=0.4, random_state=42)  # 60% train, 40% temporal
test, validation = train_test_split(temp, test_size=0.5, random_state=42)  # 50% del temporal per a test i validació


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

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def calcular_matriz_confusion_item(test, item_item_matriu, threshold=3.0):
    """
    Calcula i mostra la matriu de confusió per al recomanador item-item després de convertir a binari.

    Args:
        test (DataFrame): Dades de prova amb "User ID", "Product ID" i "Rating".
        item_item_matriu (DataFrame): Matriu item-producte amb prediccions.
        threshold (float): Llindar per considerar una valoració com a positiva.

    Returns:
        dict: Diccionari amb FP, TP, FN, TN.
    """
    # Validar IDs presents
    ids_product_test = set(test['Product ID'])
    ids_user_test = set(test['User ID'])
    ids_product_matriu = set(item_item_matriu.index)
    ids_user_matriu = set(item_item_matriu.columns)


    # Filtrar dades vàlides
    dades_valides = test[
        (test['Product ID'].isin(item_item_matriu.index)) &
        (test['User ID'].isin(item_item_matriu.columns))
    ]

    if dades_valides.empty:
        raise ValueError("No hi ha dades comunes entre el conjunt de test i la matriu item-item.")

    # Obtenir les prediccions fent servir una indexació eficient
    prediccions = dades_valides.apply(
        lambda fila: item_item_matriu.at[fila['Product ID'], fila['User ID']], axis=1
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

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def calcular_matriz_confusion_SVD(data, reconstructed_matrix, umbral=3):
  
    user_map = {u: i for i, u in enumerate(data['User ID'].unique())}
    product_map = {p: i for i, p in enumerate(data['Product ID'].unique())}

    y_real_bin = []
    y_pred_bin = []

    for _, row in data.iterrows():
        user_id = row['User ID']
        product_id = row['Product ID']
        rating = row['Rating']

        if user_id in user_map and product_id in product_map:
            user_idx = user_map[user_id]
            product_idx = product_map[product_id]

            pred_rating = reconstructed_matrix[user_idx, product_idx]
            y_real_bin.append(rating >= umbral)
            y_pred_bin.append(pred_rating >= umbral)

    tn, fp, fn, tp = confusion_matrix(y_real_bin, y_pred_bin).ravel()
    # Mostrar resultats
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)
    print("True Negatives (TN):", tn)


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


def train_svd_recommender_sparse(train_data, num_components=20):
    """
    Entrena un recomanador basat en SVD amb suport per matrius escasses.

    Args:
        train_data (DataFrame): Matriu usuari-producte (DataFrame).
        num_components (int): Nombre de components latents.

    Returns:
        svd_model (TruncatedSVD): Model entrenat.
        reconstructed_matrix (ndarray): Matriu reconstruïda amb prediccions.
    """
    # Consolidar duplicats prenent la mitjana de les valoracions
    train_data = train_data.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()

    # Crear la matriu usuari-producte
    user_product_matrix = train_data.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

    # Convertir a matriu escassa
    user_product_sparse = csr_matrix(user_product_matrix.values)
    # Entrenar el model SVD
    svd = TruncatedSVD(n_components=num_components, random_state=42)
    svd.fit(user_product_sparse)

    # Matriu latent (U * Sigma)
    latent_matrix = svd.transform(user_product_sparse)

    # Reconstruir la matriu amb les prediccions
    reconstructed_matrix = np.dot(latent_matrix, svd.components_)

    return svd, reconstructed_matrix




# USER USER
train_user = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
user_item_matriu = train_user.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

test_user = test[(test['User ID'].isin(user_item_matriu.index)) & (test['Product ID'].isin(user_item_matriu.columns))]

resultats_user = calcular_matriz_confusion_usr(test_user, user_item_matriu, threshold=3.0)
accuracy_user = calcular_accuracy(resultats_user["TP"], resultats_user["FP"], resultats_user["FN"], resultats_user["TN"])
precision_user = calcular_precision(resultats_user["TP"], resultats_user["FP"])
recall_user = calcular_recall(resultats_user["TP"], resultats_user["FN"])
f1_score_uer = calcular_f1_score(precision_user, recall_user)


# ITEM ITEM
train_item = train.groupby(['Product ID', 'User ID'], as_index=False)['Rating'].mean()
item_item_matriu = train_item.pivot(index='Product ID', columns='User ID', values='Rating').fillna(0)

test_item = test[(test['Product ID'].isin(item_item_matriu.index)) &(test['User ID'].isin(item_item_matriu.columns))]

resultats_item = calcular_matriz_confusion_item(test_item, item_item_matriu, threshold=3.0)
accuracy_item = calcular_accuracy(resultats_item["TP"], resultats_item["FP"], resultats_item["FN"], resultats_item["TN"])
precision_item = calcular_precision(resultats_item["TP"], resultats_item["FP"])
recall_item = calcular_recall(resultats_item["TP"], resultats_item["FN"])
f1_score_item = calcular_f1_score(precision_item, recall_item)


# SVD
# Entrenar el modelo SVD
svd_model, reconstructed_matrix = train_svd_recommender_sparse(data_retallat)

# Calcular la matriz de confusión
resultats_SVD = calcular_matriz_confusion_SVD(data_retallat, reconstructed_matrix, umbral=3)
accuracy_SVD = calcular_accuracy(resultats_SVD["TP"], resultats_SVD["FP"], resultats_SVD["FN"], resultats_SVD["TN"])
precision_SVD = calcular_precision(resultats_SVD["TP"], resultats_SVD["FP"])
recall_SVD = calcular_recall(resultats_SVD["TP"], resultats_SVD["FN"])
f1_score_SVD = calcular_f1_score(precision_SVD, recall_SVD)



# Función para crear las gráficas de comparación
def graficar_comparaciones(metrics, labels, title, ylabel):
    """
    Genera una gráfica de líneas comparando las métricas para los distintos recomendadores.

    Args:
        metrics (list of list): Lista de métricas para cada recomendador.
        labels (list of str): Etiquetas de los recomendadores.
        title (str): Título de la gráfica.
        ylabel (str): Etiqueta del eje Y.
    """
    plt.figure(figsize=(10, 6))
    x = range(len(labels))

    for i, metric in enumerate(metrics):
        plt.plot(x, metric, marker='o', label=f'Recommender {i+1}')

    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Recommenders')
    plt.legend()
    plt.grid(True)
    plt.show()


labels = ['User-User', 'Item-Item', 'SVD']

accuracy_values = [accuracy_user, accuracy_item, accuracy_SVD]
precision_values = [precision_user, precision_item, precision_SVD]
recall_values = [recall_user, recall_item, recall_SVD]
f1_score_values = [f1_score_uer, f1_score_item, f1_score_SVD]

graficar_comparaciones([accuracy_values], labels, 'Accuracy Comparison', 'Accuracy')
graficar_comparaciones([precision_values], labels, 'Precision Comparison', 'Precision')
graficar_comparaciones([recall_values], labels, 'Recall Comparison', 'Recall')
graficar_comparaciones([f1_score_values], labels, 'F1-Score Comparison', 'F1-Score')
