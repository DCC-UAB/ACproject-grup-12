import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


def recomanador_productes_user_user(train, usuari, numero_recomanacions=5):
    """
    Aquesta funció retorna les recomanacions per a un usuari, basant-se en la similitud amb altres usuaris.
    
    Parameters:
    - train: conjunt de dades d'entrenament amb les valoracions dels usuaris.
    - usuari: l'ID de l'usuari per al qual es volen obtenir les recomanacions.
    - numero_recomanacions: nombre de recomanacions a retornar. Per defecte els 5.
    
    Returns:
    - recomanacions: una llista dels IDs dels productes recomanats per a l'usuari.
    - similitud_coseno_df: DataFrame amb la similitud del coseno dels usuaris.
    - correlacio_pearson_df: DataFrame amb la correlació de Pearson dels usuaris.
    """
    
    # Creem la matriu d'usuari-producte a partir del conjunt d'entrenament.
    # Cada fila correspon a un usuari i cada columna a un producte, amb les valoracions com a valors.
    train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
    user_item_matrix = train.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

   
    if usuari not in user_item_matrix.index:
        print(f"L'usuari {usuari} no té valoracions a la matriu.")
        return [], None, None


    # Obtenim les valoracions de l'usuari
    valoracions_usuari = user_item_matrix.loc[usuari]

    similitud_coseno = cosine_similarity([valoracions_usuari], user_item_matrix)[0]
    similitud_coseno_df = pd.DataFrame(similitud_coseno, index=user_item_matrix.index, columns=["Similitud Coseno"])

    similitud_pearson = []
    for usuari_index in user_item_matrix.index:
        corr, _ = pearsonr(valoracions_usuari, user_item_matrix.loc[usuari_index])
        similitud_pearson.append(corr)

    correlacio_pearson_df = pd.DataFrame(similitud_pearson, index=user_item_matrix.index, columns=["Correlacio Pearson"])

    similitud_coseno_df = similitud_coseno_df.sort_values(by="Similitud Coseno", ascending=False)
    correlacio_pearson_df = correlacio_pearson_df.sort_values(by="Correlacio Pearson", ascending=False)

    similitud_coseno_df = similitud_coseno_df.drop(usuari)
    correlacio_pearson_df = correlacio_pearson_df.drop(usuari)

    # Obtenim els usuaris més similars (en aquest cas, els 5 més similars) segons la similitud del coseno
    usuaris_similars_coseno = similitud_coseno_df.head(5).index
    usuaris_similars_pearson = correlacio_pearson_df.head(5).index

    # Generem una llista de recomanacions basant-nos en els productes valorats per aquests usuaris similars
    productes_recomanats_coseno = []
    for usuari_similar in usuaris_similars_coseno:
        productes_similar = user_item_matrix.loc[usuari_similar]
        productes_similar = productes_similar[productes_similar > 0]  
        productes_recomanats_coseno.extend(productes_similar.index)

    productes_recomanats_pearson = []
    for usuari_similar in usuaris_similars_pearson:
        productes_similar = user_item_matrix.loc[usuari_similar]
        productes_similar = productes_similar[productes_similar > 0]  
        productes_recomanats_pearson.extend(productes_similar.index)


    productes_recomanats_coseno = list(set(productes_recomanats_coseno) - set(valoracions_usuari[valoracions_usuari > 0].index))
    productes_recomanats_pearson = list(set(productes_recomanats_pearson) - set(valoracions_usuari[valoracions_usuari > 0].index))

    # Retornem les primeres recomanacions (els 5 productes mÃ©s recomanats segons coseno i Pearson)
    recomanacions_coseno = productes_recomanats_coseno[:numero_recomanacions]
    recomanacions_pearson = productes_recomanats_pearson[:numero_recomanacions]

    return recomanacions_coseno, recomanacions_pearson, similitud_coseno_df, correlacio_pearson_df
