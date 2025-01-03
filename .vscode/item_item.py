
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
def recomanador_productes_item_item(data, producte_referencia, numero_recomanacions=5):
  

    # Eliminar duplicats per evitar dades inconsistents
    data_sense_duplicats = data.drop_duplicates(subset=['User ID', 'Product ID'])

    # Crear matriu Usuari-Producte (valors: Rating)
    user_item_matriu = data_sense_duplicats.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

    # Comprovació si el producte existeix
    if producte_referencia not in user_item_matriu.columns:
        print(f"El producte '{producte_referencia}' no està a les dades.")
        return [], None, None

    # Calcular la similitud entre productes amb similitud del cosinus
    similitud_productes_coseno = cosine_similarity(user_item_matriu.T)
    similitud_coseno_df = pd.DataFrame(similitud_productes_coseno, 
                                       index=user_item_matriu.columns, 
                                       columns=user_item_matriu.columns)
    
    # Calcular la correlació de Pearson entre productes amb Numpy (optimitzat)
    mean_centered = user_item_matriu - user_item_matriu.mean(axis=0)
    norm = np.linalg.norm(mean_centered, axis=0)
    normalized_matrix = mean_centered / norm
    correlacio_pearson_np = np.dot(normalized_matrix.T, normalized_matrix)

    # Convertir a DataFrame
    correlacio_pearson_df = pd.DataFrame(correlacio_pearson_np, 
                                         index=user_item_matriu.columns, 
                                         columns=user_item_matriu.columns)
    
    # Ordenar els productes més similars al producte de referència segons cosinus
    productes_similars_coseno = similitud_coseno_df[producte_referencia].sort_values(ascending=False)

    # Ordenar els productes més similars al producte de referència segons Pearson
    productes_similars_pearson = correlacio_pearson_df[producte_referencia].sort_values(ascending=False)

    # Filtrar el producte de referència i limitar al número de recomanacions
    recomanacions_coseno = productes_similars_coseno.drop(producte_referencia).head(numero_recomanacions).index.tolist()
    recomanacions_pearson = productes_similars_pearson.drop(producte_referencia).head(numero_recomanacions).index.tolist()

    # Retornar recomanacions i matrius de similitud
    return recomanacions_coseno, recomanacions_pearson, similitud_coseno_df, correlacio_pearson_df

