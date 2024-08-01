import pandas as pd
import numpy as np

# Standardisation et évaluation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

# Réduction de dimension
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Undersampling
from imblearn.under_sampling import RandomUnderSampler

# Divers
from utils import racine_projet

def prepross_reg(test_size = 0.2, scaler_type='standard', dim_reduction_method='embedded'):
    """
    Réalise le préprocessing pour un modèle de régression

    Parameters:
    ----------
    test_size : float, optionnal, default = 0.2
        La taille de l'échantillon à gardé pour tester le modèle lors du random test_split.

    scaler_type : str, optional, default='standard'
        Le type de scaler à utiliser : 'standard' pour StandardScaler, 'minmax' pour MinMaxScaler.
        
    dim_reduction_method : str, optional, default='embedded'
        La méthode de réduction de dimension à utiliser : 'lda', 'embedded' ou 'filtering'.
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = test_size, random_state=42)

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
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train[columns_for_scaling]= scaler.fit_transform(X_train[columns_for_scaling])
    X_test[columns_for_scaling] = scaler.transform(X_test[columns_for_scaling]) 

    ## 5 - réduction de dimension
    if dim_reduction_method == 'filtering': # filtering
        sel = VarianceThreshold(1e-6)
        X_train = sel.fit_transform(X_train)
        X_test = sel.transform(X_test)
    
    elif dim_reduction_method == 'lda': # lda
        lda = LDA(n_components = 0.95)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
    
    else:  # embedded method
        lasso = Lasso(alpha=1)
        model = SelectFromModel(estimator=lasso, threshold=1e-10)
        model.fit(X_train, y_train)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)
    
    return (X_train, y_train, X_test, y_test)    

def prepross_class(test_size = 0.2, sampling = 50000, scaler_type='standard', dim_reduction_method='none'):
    """
    Réalise le préprocessing pour un modèle de classification

    Parameters:
    ----------
    test_size : float, optionnal, default = 0.2
        La taille de l'échantillon à garder pour tester le modèle lors du random test_split. 

    sampling: int, optionnal, default = 50000
        La taille de l'échantillon pour chaque catégorie à conserver lors de l'undersampling.    

    scaler_type : str, optional, default='standard'
        Le type de scaler à utiliser : 'standard' pour StandardScaler, 'minmax' pour MinMaxScaler.
        
    dim_reduction_method : str, optional, default='none'
        La méthode de réduction de dimension à utiliser : 'lda', 'pca' ou 'none'.        
        
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
    X_train, X_test, y_train, y_test = train_test_split(X,target, test_size = test_size, random_state=42)

    ### 3.3 Undersampling
    sampling_strategy = {0: sampling, 1: sampling, 2: sampling, 3: sampling} # Définition du nombre de lignes souhaitées dans chaque classe d'échantillon
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
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train[columns_for_scaling]= scaler.fit_transform(X_train[columns_for_scaling])
    X_test[columns_for_scaling] = scaler.transform(X_test[columns_for_scaling]) 

    ## 5 - réduction de dimension
    ### PCA
    if dim_reduction_method == 'pca':
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    ### LDA
    elif dim_reduction_method == 'lda':
        lda = LDA(n_components=min(X_train.shape[1], len(set(y_train)) - 1))
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
    
    ### none
    else :
        X_train = X_train 
        X_test = X_test

    return (X_train, y_train, X_test, y_test)    
    