import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Funció optimitzada per calcular el MAE per item-item
def calcular_mae_item_item(test, matriu):
    """
    Calcula el MAE entre les valoracions reals i les prediccions per al recomanador item-item.
    """

    # Normalització de les valoracions
    min_rating = test['Rating'].min()
    max_rating = test['Rating'].max()
    test['Rating'] = (test['Rating'] - min_rating) / (max_rating - min_rating)

    # Filtrar productes/usuaris poc valorats
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

    # Filtrar dades vàlides
    dades_valides = test[
        test['Product ID'].isin(matriu.index) &   # Canvia 'User ID' per 'Product ID'
        test['User ID'].isin(matriu.columns)     # Canvia 'Product ID' per 'User ID'
    ]

    if dades_valides.empty:
        raise ValueError("El conjunt de dades de prova no té valors comuns amb la matriu producte-usuari.")

    # Obtenir prediccions de manera vectoritzada
    prediccions = dades_valides.apply(
        lambda fila: matriu.at[fila['Product ID'], fila['User ID']], axis=1  # Canvia l'ordre d'accés
    )

    valoracions_reals = dades_valides['Rating'].values
    prediccions = prediccions.values
    mascara = ~np.isnan(prediccions)
    mae = mean_absolute_error(valoracions_reals[mascara], prediccions[mascara])
    
    return mae

# Funció per simular el càlcul de MAE en diferents iteracions
def simular_mae_iteracions_item_item(test, product_user_matriu, iteracions=[50, 100, 250, 500], num_simulacions=10):
    """
    Simula càlculs de MAE diverses vegades amb diferents iteracions per a recomanador item-item.
    """
    resultats = {}
    
    for num_iteracions in iteracions:
        valors_mae = []
        for _ in range(num_simulacions):
            matriu_amb_soroll = product_user_matriu + np.random.normal(0, 0.1, product_user_matriu.shape)
            matriu_actual = pd.DataFrame(matriu_amb_soroll, 
                                         index=product_user_matriu.index, 
                                         columns=product_user_matriu.columns)
            mae = calcular_mae_item_item(test, matriu_actual)
            valors_mae.append(mae)
        
        valor_mitja_mae = np.mean(valors_mae) 
        resultats[num_iteracions] = valor_mitja_mae
        print(f"Per la iteració: {num_iteracions}, el valor mitjà del MAE és: {valor_mitja_mae:.4f}")
    
    return resultats

def dibuixar_grafic_mae_item_item(resultats_mae):
    """
    Mostra un gràfic de línia amb els MAE obtinguts per al recomanador item-item.
    """
    plt.figure(figsize=(8, 6))
    iteracions = list(resultats_mae.keys())
    mae_values = list(resultats_mae.values())
    
    plt.plot(iteracions, mae_values, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel("Número d'Iteracions")
    plt.ylabel("MAE")
    plt.title("Evolució del MAE segons les Iteracions (Item-Item)")
    plt.grid(True)

    plt.show(block=False)  
    input("Prem Enter per continuar...") 
