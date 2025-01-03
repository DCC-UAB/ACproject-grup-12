import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.getcwd())


from user_user import recomanador_productes_user_user
from item_item import recomanador_productes_item_item

from Error_rmse_user import simular_rmse_iteracions_user_user, dibuixar_grafic_rmse_user_user
from Error_rmse_item import simular_rmse_iteracions_item_item, dibuixar_grafic_rmse_item_item
from Error_mae_user import simular_mae_iteracions_user_user, dibuixar_grafic_mae_user_user
from Error_mae_item import simular_mae_iteracions_item_item, dibuixar_grafic_mae_item_item

from errorsSVD import train_svd_recommender_sparse, recommend_top_n
from plots import distribucio, correlacio 


sys.path.append(os.getcwd())

# Carregar el datasets
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

# Crear la gràfica
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

info = input ("Vols veure informació del Dataset abans i després de netejar-lo? (Si/No) ")

if info == "Si":
    total_ratings_2 = data_retallat.shape[0]
    total_usuaris_2 = data_retallat['User ID'].nunique()
    total_productes_2 = data_retallat['Product ID'].nunique()

    print("Nombre de usuaris abans de netejar el dataset: ", total_usuaris_1)
    print("Nombre de usuaris després de netejar el dataset: ", total_usuaris_2)
    print("Nombre de productes abans de netejar el dataset: ", total_productes_1)
    print("Nombre de productes després de netejar el dataset: ", total_productes_2)
    print("Nombre de ratings abans de netejar el dataset: ", total_ratings_1)
    print("Nombre de ratings després de netejar el dataset: ", total_ratings_2)
    # Abans de netejar
    axes[0].bar(rating_counts_before.index, rating_counts_before.values, color='lightblue')
    axes[0].set_title('Distribució de les Valoracions Abans de Netejar')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Quantitat d\'usuaris')

    # Després de netejar
    axes[1].bar(rating_counts_after.index, rating_counts_after.values, color='lightgreen')
    axes[1].set_title('Distribució de les Valoracions Després de Netejar')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Quantitat d\'usuaris')

    plt.tight_layout()
    plt.show(block=False)  # Mostra el gràfic sense bloquejar l'execució
    input("Prem Enter per continuar...")  # Manté la finestra oberta fins que es prem Enter



# Dividir el conjunt en train (60%), test (20%) i validació (20%)
train, temp = train_test_split(data_retallat, test_size=0.4, random_state=42)  # 60% train, 40% temporal
test, validation = train_test_split(temp, test_size=0.5, random_state=42)  # 50% del temporal per a test i validació

train.to_csv('train_dataset.csv', index=False)
temp.to_csv('temporal_dataset.csv', index=False)
test.to_csv('test_dataset.csv', index=False)
validation.to_csv('validation_dataset.csv', index=False)

# gràfiques del dataset de correlació i distribució
plots = str(input("Vols mostrar gràfics de distribucio i correlació: (Si/No) "))
if plots == "Si":
    distribucio(test)
    correlacio(test)





# Sol·licitar el tipus de recomanació
tipo = str(input("Vols recomanar per 'user_user' o 'item_item' o 'SVD': "))

# Comprovar quantes valoracions ha fet un usuari abans de continuar
if tipo == "user_user":
    while True:
        usuari = str(input("Introdueix l'usuari pel qual vols fer recomanacions: "))
        
        # Comprovar quantes valoracions ha fet l'usuari
        num_valoracions_usuari = len(data_retallat[data_retallat['User ID'] == usuari])
        print(f"L'usuari {usuari} ha fet {num_valoracions_usuari} valoracions.")
        
        # Comprovació si l'usuari té almenys 10 valoracions
        if num_valoracions_usuari >= 10:
            # Si l'usuari té més de 10 valoracions, es poden generar recomanacions
            recomanacions_coseno, recomanacions_pearson, similitud_coseno_df, correlacio_pearson_df = recomanador_productes_user_user(train, usuari, numero_recomanacions=5)
            
            tipo_rec = int(input("Vols recomanar per Similaritat de Cosinus (1), per Correlació de Pearson (2)? "))
            # Imprimir les recomanacions
            if tipo_rec == 1:
                print(f"Recomanacions per a l'usuari {usuari} basades en Similaritat de Cosinus: ")
                print(recomanacions_coseno)
                recomanacions = recomanacions_coseno
            if tipo_rec == 2:
                print(f"Recomanacions per a l'usuari {usuari} basades en correlació de Pearson: ")
                print(recomanacions_pearson)
                recomanacions = recomanacions_pearson


            # Calcular l'error RMSE i MAE
            train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
            user_item_matriu = train.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

            calcular_error_RMSE = str(input("Vols calcular l'error RMSE per al model user-user? (Si/No): "))
            if calcular_error_RMSE == "Si":

                # Simular càlculs de RMSE amb diferents iteracions
                resultats_rmse = simular_rmse_iteracions_user_user(test, user_item_matriu)
                # Generar el boxplot per mostrar els resultats
                dibuixar_grafic_rmse_user_user(resultats_rmse)

            calcular_error_MAE = str(input("Vols calcular l'error MAE per al model user-user? (Si/No): "))
            if calcular_error_MAE == "Si":  
                # Simular càlculs de MAE
                resultats_mae = simular_mae_iteracions_user_user(test, user_item_matriu)
                
                # Mostrar el boxplot
                dibuixar_grafic_mae_user_user(resultats_mae)

        
        else:
            # Si l'usuari té menys de 10 valoracions, mostrar un missatge i continuar demanant
            print(f"L'usuari {usuari} té menys de 10 valoracions. Si us plau, introdueix un altre.")
        
       
if tipo == "item_item":
    producte_referencia = str(input("Introdueix l'ID del producte: "))

    recomanacions_coseno, recomanacions_pearson, similitud_coseno_df, correlacio_pearson_df = recomanador_productes_item_item(train, producte_referencia, numero_recomanacions=5)

    tipo_rec = int(input("Vols recomanar per Similaritat de Cosinus (1), per Correlació de Pearson (2)? "))
    # Imprimir les recomanacions
    if tipo_rec == 1:
        print(f"Recomanacions segons el producte {producte_referencia} basades en Similaritat de Cosinus: {recomanacions_coseno}")
        recomanacions = recomanacions_coseno
    if tipo_rec == 2:
        print(f"Recomanacions segons el producte {producte_referencia} basades en correlació de Pearson: {recomanacions_pearson}")
        recomanacions = recomanacions_pearson

    # Comprovar quantes valoracions ha fet l'usuari
    valoracions_usuari = train[train['Product ID'] == producte_referencia]
    valoracions_per_usuari = valoracions_usuari.groupby('User ID').size()


    # Càlcul d'errors
    train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()
    item_user_matriu = train.pivot(index='Product ID', columns='User ID', values='Rating').fillna(0)

    # Calculant errors RMSE i MAE
    calcular_error_RMSE = str(input("Vols calcular l'error RMSE per al model item-item? (Si/No): "))
    if calcular_error_RMSE == "Si":
        # Simular càlculs de RMSE amb diferents iteracions
        resultats_rmse = simular_rmse_iteracions_item_item(test, item_user_matriu)
        # Generar el boxplot per mostrar els resultats
        dibuixar_grafic_rmse_item_item(resultats_rmse)


    calcular_error_MAE = str(input("Vols calcular l'error MAE per al model item-item? (Si/No): "))
    if calcular_error_MAE == "Si":
        resultats_mae = simular_mae_iteracions_item_item(test, item_user_matriu)
        dibuixar_grafic_mae_item_item(resultats_mae)   
 
   


if tipo == "SVD":
    
    train = train.groupby(['User ID', 'Product ID'], as_index=False)['Rating'].mean()

    svd_model, reconstructed_matrix = train_svd_recommender_sparse(train, num_components=10)
    usuari = str(input("Introdueix l'usuari pel qual vols fer recomanacions: "))
    
    try:
        top_recommendations = recommend_top_n(usuari, train, reconstructed_matrix, n=5)
        print(f"Top 5 recomanacions per a l'usuari {usuari}:")
        for product_id, score in top_recommendations:
            print(f"Producte: {product_id}, Puntuació estimada: {score:.2f}")
    except ValueError as e:
        print(e)


  
  
