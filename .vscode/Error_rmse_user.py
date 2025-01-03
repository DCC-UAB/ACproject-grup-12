import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def calcular_rmse_user_user(test, matriu):
    """
    Calcula el RMSE de manera vectoritzada entre les dades de prova i la matriu usuari-producte.
    Inclou:
    - Normalització de valoracions
    - Eliminació de valors extrems
    - Filtratge d'usuaris i productes poc valorats
    """
    # Normalització de les valoracions
    min_rating = test['Rating'].min()
    max_rating = test['Rating'].max()
    test['Rating'] = (test['Rating'] - min_rating) / (max_rating - min_rating)

    # Filtrar usuaris/productes poc valorats
    min_valoracions = 5  
    user_counts = test['User ID'].value_counts()
    product_counts = test['Product ID'].value_counts()
    
    test = test[test['User ID'].isin(user_counts[user_counts >= min_valoracions].index)]
    test = test[test['Product ID'].isin(product_counts[product_counts >= min_valoracions].index)]

    # Eliminació de valors extrems (outliers)
    mean_rating = test['Rating'].mean()
    std_rating = test['Rating'].std()
    test = test[(test['Rating'] >= mean_rating - 3 * std_rating) &
                (test['Rating'] <= mean_rating + 3 * std_rating)]

    # Filtrar dades vàlides per a l'RMSE
    dades_valides = test[
        test['User ID'].isin(matriu.index) &
        test['Product ID'].isin(matriu.columns)
    ]

    if dades_valides.empty:
        raise ValueError("El conjunt de dades de prova no té valors comuns amb la matriu usuari-producte.")

    # Obtenir les prediccions fent servir una indexació eficient
    prediccions = dades_valides.apply(
        lambda fila: matriu.at[fila['User ID'], fila['Product ID']], axis=1)

    valoracions_reals = dades_valides['Rating'].values

    prediccions = prediccions.values
    mascara = ~np.isnan(prediccions)  
    rmse = np.sqrt(mean_squared_error(valoracions_reals[mascara], prediccions[mascara]))
    return rmse



def simular_rmse_iteracions_user_user(dades_prova, matriu_usuari_producte, iteracions=[50, 100, 250, 500]):
    """
    Simula càlculs de RMSE diverses vegades amb diferents iteracions per generar dades per a un boxplot.
    Inclou:
    - Normalització de valoracions
    - Eliminació de valors extrems
    - Filtratge d'usuaris i productes poc valorats
    """
    resultats = {}  # Diccionari per emmagatzemar els resultats de RMSE per cada iteració

    # Iterar per cada número d'iteracions
    for num_iteracions in iteracions:
        valors_rmse = []  
        for _ in range(10):
            matriu_amb_soroll = matriu_usuari_producte + np.random.normal(0, 0.05, matriu_usuari_producte.shape)

            rmse = calcular_rmse_user_user(dades_prova, pd.DataFrame(
                matriu_amb_soroll, 
                index=matriu_usuari_producte.index, 
                columns=matriu_usuari_producte.columns
            ))
            valors_rmse.append(rmse)

        valor_mitja_rmse = np.mean(valors_rmse)
        resultats[num_iteracions] = valor_mitja_rmse  
        print(f"Per la iteració: {num_iteracions}, el valor mitjà del RMSE és: {valor_mitja_rmse:.4f}")

    return resultats

def dibuixar_grafic_rmse_user_user(resultats_rmse):

    plt.figure(figsize=(8, 6))
    iteracions = list(resultats_rmse.keys())
    rmse_values = list(resultats_rmse.values())

    plt.plot(iteracions, rmse_values, marker='o', linestyle='-', linewidth=2, markersize=6)

    plt.xlabel("Número d'Iteracions")
    plt.ylabel("RMSE")
    plt.title("Evolució del RMSE segons les Iteracions")
    plt.grid(True)
  
    plt.show(block=False)  
    input("Prem Enter per continuar...") 
    




