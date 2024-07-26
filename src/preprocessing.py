import pandas as pd
import numpy as np

#Pour la standardisation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#Pour l'undersampling
from imblearn.under_sampling import RandomUnderSampler

#Pour la réduction de dimension
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils import racine_projet


def prepross_reg():
    """
    Réalise le préprocessing pour un modèle de régression

    Parameters:
    ----------
    Pas de paramètres à renseigner.
    

    Returns:
    -------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).
    """
    
    df = pd.read_csv(racine_projet()+'/data/processed/ML_data.csv', low_memory = False)
    
    ## 1 - Filtre rajoutés
    df = df.drop(columns = ["DeployedFromStation_Name"]) # suppression de la variable DeployedFromStation_Name suite à son impractibilité (120 stations différentes) et redondance avec la localisation en un un sens

    ## 2 - encodage
    df_encoded = df  
    ### 2.1 Encodage binaire : DeployedFromLocation
    encoder = LabelEncoder() 
    df_encoded['DeployedFromLocation'] = encoder.fit_transform(df_encoded['DeployedFromLocation'])

    ### 2.2 Encodage getdummies : "PlusCode_Description", "PropertyCategory","AddressQualifier","IncidentType"
    cols_to_encode = df_encoded.select_dtypes(include=['object']).columns 
    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode)
    
    ## 3 - train_test_split
    target = df_encoded.ResponseDuration
    X = df_encoded.drop(labels = ["ResponseDuration"], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=42)

    ## 4 - normalisation (Standard OU Minmax)
    columns_for_scaling = ["Northing_rounded",
                           "Easting_rounded",
                           "PumpOrder",
                           "NumStationsWithPumpsAttending",
                           "NumPumpsAttending",
                           "NumCalls",
                           "year",
                           "rain", # rajoutée
                           "temperature_2m",  # rajoutée
                           "relative_humidity_2m", # rajoutée
                           "weather_code", # rajoutée
                           "wind_speed_10m", # rajoutée
                           "wind_gusts_10m"] # rajoutée
    scaler = StandardScaler()
    #scaler = MinMaxScaler()

    X_train[columns_for_scaling]= scaler.fit_transform(X_train[columns_for_scaling])
    X_test[columns_for_scaling] = scaler.transform(X_test[columns_for_scaling]) 

    ## 5 - Réduction de dimension
    ### 5.1 - Filtering method
    #sel = VarianceThreshold(1e-6)
    #X_train_fit = sel.fit_transform(X_train)
    #X_test_fit = sel.transform(X_test)
    #X_train = X_train_fit
    #X_test = X_test_fit

    ### 5.2 - Embedded method
    lasso = Lasso(alpha = 1)
    model = SelectFromModel(estimator = lasso, threshold = 1e-10)
    model.fit(X_train, y_train)

    X_train_emb = model.fit_transform(X_train, y_train)
    X_test_emb = model.transform(X_test)
    X_train = X_train_emb
    X_test = X_test_emb

    ### 5.3 - PCA
    #pca = PCA(n_components = 0.95)
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)
    #pca.fit(X_train)
    #X_train = X_train_pca
    #X_test = X_test_pca
    
    return (X_train, y_train, X_test, y_test)    

def prepross_class():
    """
    Réalise le préprocessing pour un modèle de classification

    Parameters:
    ----------
    Pas de paramètres à renseigner.
    

    Returns:
    -------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).
    """
    df = pd.read_csv(racine_projet()+'/data/processed/ML_data.csv', low_memory = False)
    
    ## 1 - Filtre rajoutés
    df = df.drop(columns = ["DeployedFromStation_Name"]) # suppression de la variable DeployedFromStation_Name suite à son impractibilité (120 stations différentes) et redondance avec la localisation en un un sens

    ## 2 - encodage
    df_encoded = df  
    ### 2.1 Encodage binaire : DeployedFromLocation
    encoder = LabelEncoder() 
    df_encoded['DeployedFromLocation'] = encoder.fit_transform(df_encoded['DeployedFromLocation'])

    ### 2.2 Encodage getdummies : "PlusCode_Description", "PropertyCategory","AddressQualifier","IncidentType"
    cols_to_encode = df_encoded.select_dtypes(include=['object']).columns 
    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode)
    
    ## 3 - train_test_split
    
    ### 3.1 Catégorisation de la variable target
    bins = [df_encoded['ResponseDuration'].min(), 251, 325, 421, df_encoded['ResponseDuration'].max()] # définition des catégories (ici selon les quartiles)
    labels = [0,1,2,3]
    df_encoded['ResponseCat'] = pd.cut(df_encoded.ResponseDuration, bins=bins, labels= labels, include_lowest=True) # découpage de la target
    df_encoded = df_encoded.drop(columns = ['ResponseDuration']) # suppression de la variable non-catégorisée

    ### 3.2 train_test_split
    target = df_encoded['ResponseCat'] # définition de la target sur la variable catégorisée
    X = df_encoded.drop(labels = ["ResponseCat"], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=42)

    ### 3.3 Undersampling
    sampling_strategy = {0: 50000, 1: 50000, 2: 50000, 3: 50000} # Définition du nombre de lignes souhaitées dans chaque classe d'échantillon
    rus = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    
    ## 4 - normalisation (Standard OU Minmax)
    columns_for_scaling = ["Northing_rounded",
                           "Easting_rounded",
                           "PumpOrder",
                           "NumStationsWithPumpsAttending",
                           "NumPumpsAttending",
                           "NumCalls",
                           "year",
                           "rain", # rajoutée
                           "temperature_2m",  # rajoutée
                           "relative_humidity_2m", # rajoutée
                           "weather_code", # rajoutée
                           "wind_speed_10m", # rajoutée
                           "wind_gusts_10m"] # rajoutée
    scaler = StandardScaler()
    #scaler = MinMaxScaler()

    X_train[columns_for_scaling]= scaler.fit_transform(X_train[columns_for_scaling])
    X_test[columns_for_scaling] = scaler.transform(X_test[columns_for_scaling]) 

    ## 5 - réduction de dimension
    
    ### 5.1 - LDA
    #lda = LDA()
    #X_train_lda = lda.fit_transform(X_train, y_train)
    #X_test_lda = lda.transform(X_test)
    #X_train = X_train_lda
    #X_test = X_test_lda

    ### 5.2 - PCA
    #pca = PCA(n_components = 0.95)
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)
    #pca.fit(X_train)
    #X_train = X_train_pca
    #X_test = X_test_pca
    
    return (X_train, y_train, X_test, y_test)    
    