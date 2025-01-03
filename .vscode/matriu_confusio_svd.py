from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
 
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)
    print("True Negatives (TN):", tn)

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
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title("Confusion Matrix - SVD Recommender")
    plt.show()

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

# Neteja del dataset
# Eliminar valors nuls i validar el rang de valoracions
data = data.dropna()  # Elimina les files amb valors nuls
data = data[(data['Rating'] >= 1) & (data['Rating'] <= 5)]  # Filtra les classificacions invàlides

# Filtratge dels usuaris amb almenys 10 valoracions
usuaris_amb_mes_de_10_reviews = data.groupby('User ID').size()

# Filtrar els usuaris amb almenys 10 valoracions
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

# Entrenar el modelo SVD
svd_model, reconstructed_matrix = train_svd_recommender_sparse(data_retallat)

# Calcular la matriz de confusión
resultats = calcular_matriz_confusion_SVD(data_retallat, reconstructed_matrix, umbral=3)
accuracy = calcular_accuracy(resultats["TP"], resultats["FP"], resultats["FN"], resultats["TN"])
precision = calcular_precision(resultats["TP"], resultats["FP"])
recall = calcular_recall(resultats["TP"], resultats["FN"])
f1_score = calcular_f1_score(precision, recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

