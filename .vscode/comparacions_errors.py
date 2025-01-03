import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.calibration import LabelEncoder
sys.path.append(os.getcwd())

from user_user import recomanador_productes_user_user
from item_item import recomanador_productes_item_item

from Error_rmse_user import simular_rmse_iteracions_user_user
from Error_rmse_item import calcular_rmse_item_item, simular_rmse_iteracions_item_item
from Error_mae_user import calcular_mae_user_user, simular_mae_iteracions_user_user
from Error_mae_item import simular_mae_iteracions_item_item

from errorsSVD import train_svd_recommender_sparse,recommend_top_n





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

# Distribució de les valoracions abans de netejar el dataset
rating_counts_before = data['Rating'].value_counts().sort_index()
data = data.dropna() 
data = data[(data['Rating'] >= 1) & (data['Rating'] <= 5)]  # Filtra les classificacions invàlides
usuaris_amb_mes_de_10_reviews = data.groupby('User ID').size()
usuaris_adequats = usuaris_amb_mes_de_10_reviews[usuaris_amb_mes_de_10_reviews >= 10].index
data_filtrada = data[data['User ID'].isin(usuaris_adequats)]
productes_amb_mes_de_2_reviews = data_filtrada.groupby('Product ID').size()
productes_adequats = productes_amb_mes_de_2_reviews[productes_amb_mes_de_2_reviews >= 2].index
data_retallat = data_filtrada[data_filtrada['Product ID'].isin(productes_adequats)]
rating_counts_after = data_filtrada['Rating'].value_counts().sort_index()
data_retallat = data_retallat[['User ID', 'Product ID', 'Rating']]

train, temp = train_test_split(data_retallat, test_size=0.4, random_state=42)  # 60% train, 40% temporal
test, validation = train_test_split(temp, test_size=0.5, random_state=42)  # 50% del temporal per a test i validació


# USER-USER
usuari = str(input("Introdueix l'usuari pel qual vols fer recomanacions: "))

num_valoracions_usuari = len(data_retallat[data_retallat['User ID'] == usuari])
print(f"L'usuari {usuari} ha fet {num_valoracions_usuari} valoracions.")

if num_valoracions_usuari >= 10:
    recomanacions_coseno, recomanacions_pearson, similitud_coseno_df, correlacio_pearson_df = recomanador_productes_user_user(train, usuari, numero_recomanacions=5)
    
    tipo_rec = int(input("Vols recomanar per Similaritat de Cosinus (1), per Correlació de Pearson (2)? "))

    if tipo_rec == 1:
        print(f"Recomanacions per a l'usuari {usuari} basades en Similaritat de Cosinus: {recomanacions_coseno}")
        recomanacions = recomanacions_coseno
    if tipo_rec == 2:
        print(f"Recomanacions per a l'usuari {usuari} basades en correlació de Pearson: {recomanacions_pearson}")
        recomanacions = recomanacions_pearson

    train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
    user_item_matriu = train.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

    resultats_rmse_user_user = simular_rmse_iteracions_user_user(test, user_item_matriu)
    resultats_mae_user_user = simular_mae_iteracions_user_user(test, user_item_matriu)

    

# ITEM-ITEM
producte_referencia = str(input("Introdueix l'ID del producte: "))
recomanacions_coseno, recomanacions_pearson, similitud_coseno_df, correlacio_pearson_df = recomanador_productes_item_item(train, producte_referencia, numero_recomanacions=5)

tipo_rec = int(input("Vols recomanar per Similaritat de Cosinus (1), per Correlació de Pearson (2)? "))

if tipo_rec == 1:
    print(f"Recomanacions segons el producte {producte_referencia} basades en Similaritat de Cosinus: {recomanacions_coseno}")
    recomanacions = recomanacions_coseno
if tipo_rec == 2:
    print(f"Recomanacions segons el producte {producte_referencia} basades en correlació de Pearson: {recomanacions_pearson}")
    recomanacions = recomanacions_pearson

valoracions_usuari = train[train['Product ID'] == producte_referencia]
valoracions_per_usuari = valoracions_usuari.groupby('User ID').size()


item_user_matriu = train.pivot(index='Product ID', columns='User ID', values='Rating').fillna(0)

resultats_rmse_item_item = simular_rmse_iteracions_item_item(test, item_user_matriu)
resultats_mae_item_item = simular_mae_iteracions_item_item(test, item_user_matriu)


# SVD
train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
svd_model, reconstructed_matrix = train_svd_recommender_sparse(train, num_components=10)

num_components_list = [50, 100, 250, 500]
resultats_rmse_svd = calcular_rmse_per_iteracions_svd(train, test, num_components_list)
resultats_mae_svd = calcular_mae_per_iteracions_svd(train, test, num_components_list)



# Funció per a mostrar els errors RMSE dels tres recomanadors
def plot_rmse_comparison(resultats_rmse_user_user, resultats_rmse_item_item, resultats_rmse_svd, iteracions):
   
    plt.figure(figsize=(10, 6))

    plt.plot(iteracions, list(resultats_rmse_user_user.values()), marker='o', label="User-User")
    plt.plot(iteracions, list(resultats_rmse_item_item.values()), marker='s', label="Item-Item")
    plt.plot(iteracions, list(resultats_rmse_svd.values()), marker='^', label="SVD")

    plt.xlabel("Número d'Iteracions")
    plt.ylabel("RMSE")
    plt.title("Comparació de RMSE entre recomanadors")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mae_comparison(resultats_mae_user_user, resultats_mae_item_item, resultats_mae_svd, iteracions):
   
    plt.figure(figsize=(10, 6))

    plt.plot(iteracions, list(resultats_mae_user_user.values()), marker='o', label="User-User")
    plt.plot(iteracions, list(resultats_mae_item_item.values()), marker='s', label="Item-Item")
    plt.plot(iteracions, list(resultats_mae_svd.values()), marker='^', label="SVD")

    plt.xlabel("Número d'Iteracions")
    plt.ylabel("MAE")
    plt.title("Comparació de MAE entre recomanadors")
    plt.legend()
    plt.grid(True)
    plt.show()


iteracions_svd = num_components_list 


# RMSE
plot_rmse_comparison(resultats_rmse_user_user, resultats_rmse_item_item, resultats_rmse_svd, iteracions_svd)

# MAE
plot_mae_comparison(resultats_mae_user_user, resultats_mae_item_item, resultats_mae_svd, iteracions_svd)