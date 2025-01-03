import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def distribucio(test):
    plt.figure(figsize=(10, 6))
    sns.histplot(test['Rating'], bins=5, kde=True)
    plt.title('Distribució de les Valoracions')
    plt.xlabel('Valoració')
    plt.ylabel('Freqüència')
    plt.show(block=False)  # Mostra el gràfic sense bloquejar l'execució
    input("Prem Enter per continuar...") 


def correlacio(test):

    product_ratings = test.groupby('Product ID')['Rating'].mean()  # Valoració mitjana per producte
    product_ratings_count = test.groupby('Product ID')['Rating'].count()  # Nombre de valoracions per producte

    plt.figure(figsize=(12, 8))
    plt.scatter(product_ratings, product_ratings_count, alpha=0.6)
    
    plt.title('Rellevància de la Valoració Principal vs. Nombre de Valoracions')
    plt.xlabel('Valoració Principal')
    plt.ylabel('Nombre de Valoracions')

    plt.show(block=False)  
    input("Prem Enter per continuar...") 