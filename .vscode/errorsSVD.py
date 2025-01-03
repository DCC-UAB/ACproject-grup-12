from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


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

    latent_matrix = svd.transform(user_product_sparse)

    # Reconstruir la matriu amb les prediccions
    reconstructed_matrix = np.dot(latent_matrix, svd.components_)

    return svd, reconstructed_matrix



def predict_rating(user_id, product_id, train_data, reconstructed_matrix):
    """
    Prediu la valoració per a un usuari i producte donats.

    Args:
        user_id (str): ID de l'usuari.
        product_id (str): ID del producte.
        train_data (DataFrame): Dades d'entrenament amb 'User ID', 'Product ID', 'Rating'.
        reconstructed_matrix (ndarray): Matriu reconstruïda amb prediccions.

    Returns:
        float: Predicció de la valoració.
    """
    # Mapar User ID i Product ID a les files i columnes de la matriu

    user_map = {u: i for i, u in enumerate(train_data['User ID'].unique())}
    product_map = {p: i for i, p in enumerate(train_data['Product ID'].unique())}


    if user_id in user_map and product_id in product_map:
        user_idx = user_map[user_id]
        product_idx = product_map[product_id]
        return reconstructed_matrix[user_idx, product_idx]
    else:
        return np.nan  # Si no es pot predir, torna NaN


def recommend_top_n(user_id, train_data, reconstructed_matrix, n=5):
    """
    Genera les top-N recomanacions per a un usuari donat.

    Args:
        user_id (str): ID de l'usuari.
        train_data (DataFrame): Dades d'entrenament amb 'User ID', 'Product ID', 'Rating'.
        reconstructed_matrix (ndarray): Matriu reconstruïda amb prediccions.
        n (int): Nombre de recomanacions a retornar.

    Returns:
        list: Llista de tuples amb els IDs dels productes i la puntuació estimada.
    """
    
    # Mapar User ID i Product ID a les files i columnes de la matriu
    user_map = {u: i for i, u in enumerate(train_data['User ID'].unique())}
    product_map = {p: i for i, p in enumerate(train_data['Product ID'].unique())}

    # Comprovar si l'usuari està al mapa
    if user_id not in user_map:
        raise ValueError(f"L'usuari {user_id} no està al conjunt d'entrenament.")

    user_idx = user_map[user_id]

    # Obtenir productes valorats per l'usuari
    rated_products = train_data[train_data['User ID'] == user_id]['Product ID'].unique()

    # Generar prediccions per a tots els productes
    recommendations = []
    for product_id, product_idx in product_map.items():
        if product_id not in rated_products:  # Només considerar productes no valorats
            pred_rating = reconstructed_matrix[user_idx, product_idx]
            if pred_rating > 0:  # Evitar recomanacions amb puntuacions molt baixes
                recommendations.append((product_id, pred_rating))

    # Ordenar recomanacions per puntuació descendent
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    return recommendations[:n]  # Retornar les top-N recomanacions


