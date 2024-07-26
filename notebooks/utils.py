import pandas as pd
import numpy as np
import os
import pickle

def dataframe_info(df):
    """
    Retourne une description claire du DataFrame pandas mis en paramètre permettant une analyse plus claire que .info()
    """
    # Initialiser une liste pour stocker les informations
    info_list = []

    # Parcourir chaque colonne pour recueillir les informations
    for col in df.columns:
        non_null_count = df[col].notnull().sum()
        dtype = df[col].dtype
        nan_count = df[col].isna().sum()
        nan_percentage = (nan_count / len(df)) * 100
        example_value = df[col].dropna().iloc[0] if non_null_count > 0 else None
        info_list.append([col, non_null_count, nan_count, round(nan_percentage, 2), dtype, example_value])

    # Créer un DataFrame à partir de la liste d'informations
    info_df = pd.DataFrame(info_list, columns=['Column', 'Non-Null Count', 'NaN Count', 'NaN Percentage', 'Dtype', 'Example Value'])
    
    return info_df


def racine_projet():
    """
    Retourne le chemin absolu de la racine du projet en remontant de deux niveaux.
    """
    # Obtenir le chemin absolu du répertoire où se trouve ce fichier
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    # Remonter de deux niveaux pour atteindre la racine du projet
    racine = os.path.dirname(dossier_courant)
    return racine

def save_model(model, model_name)
    """
    Renseigner le modèle et le nom que l'on souhaite lui associé pour le sauvegarder directement dans le dossier models.
    """
    chemin_fichier = racine_projet()+'/models/'+ model_name
    with open(chemin_fichier, 'wb') as file:
        pickle.dump(model, file)

def load_model(model_name)
    """
    Renseigner le nom du modèle pour le charger.
    Renvoie en sortie le modèle chargé
    """
    chemin_fichier = racine_projet()+'/models/'+ model_name

    with open(chemin_fichier, 'rb') as file:
        loaded_model = pickle.load(file)
    
    return loaded_model