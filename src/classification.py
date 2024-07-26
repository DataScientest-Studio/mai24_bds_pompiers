from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def knn_class(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de de classification KNeighborsClassifier et évalue ses performances.

    Parameters:
    ----------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).

    Returns:
    -------
    model : KNeighborsClassifier
        Le modèle de classification KNN entraîné.

    accuracy : float
        La précision du modèle sur les données de test.

    report : str
        Le rapport de classification détaillant les performances du modèle.

    cm : array
        La matrice de confusion du modèle sur les données de test.
    """
    
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_knn)
    cl_rep = classification_report(y_test, y_pred_knn)
    cm = confusion_matrix(y_test, y_pred_knn)
    
    print("Accuracy:", accuracy)
    print(cl_rep)
    print("\nConfusion Matrix:")
    print(cm)
    
    return (knn, accuracy, cl_rep, cm)

def decision_tree_class(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de classification Decision Tree et évalue ses performances.

    Parameters:
    ----------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).

    Returns:
    -------
    model : Decision Tree
        Le modèle de classification Decision Tree entraîné.

    accuracy : float
        La précision du modèle sur les données de test.

    report : str
        Le rapport de classification détaillant les performances du modèle.

    cm : array
        La matrice de confusion du modèle sur les données de test.
    """
    from sklearn.tree import DecisionTreeClassifier
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_dt)
    cl_rep = classification_report(y_test, y_pred_dt)
    cm = confusion_matrix(y_test, y_pred_dt)
    
    print("Accuracy:", accuracy)
    print(cl_rep)
    print("\nConfusion Matrix:")
    print(cm)

    return (dt, accuracy, cl_rep, cm)

def random_forest_class(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de classification Random Forest et évalue ses performances.

    Parameters:
    ----------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).

    Returns:
    -------
    model : RandomForestClassifier
        Le modèle de classification Random Forest entraîné.

    accuracy : float
        La précision du modèle sur les données de test.

    report : str
        Le rapport de classification détaillant les performances du modèle.

    cm : array
        La matrice de confusion du modèle sur les données de test.
    """

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_rf)
    cl_rep = classification_report(y_test, y_pred_rf)
    cm = confusion_matrix(y_test, y_pred_rf)
    
    print("Accuracy:", accuracy)
    print(cl_rep)
    print("\nConfusion Matrix:")
    print(cm)

    return (rf, accuracy, cl_rep, cm)


def xgb_class(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de classification XGBoost et évalue ses performances.

    Parameters:
    ----------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).

    Returns:
    -------
    model : XGBClassifier
        Le modèle de classification XGBoost entraîné avec les paramètres suivants :
        n_estimators=100,  # Nombre d'arbres à construire
        max_depth=3,        # Profondeur maximale de chaque arbre
        learning_rate=0.1,  # Taux d'apprentissage
        random_state=42     # Seed pour la reproductibilité

    accuracy : float
        La précision du modèle sur les données de test.

    report : str
        Le rapport de classification détaillant les performances du modèle.

    cm : array
        La matrice de confusion du modèle sur les données de test.
    """
    from xgboost import XGBClassifier
    
    xgb = XGBClassifier(
        n_estimators=100,  # Nombre d'arbres à construire
        max_depth=3,        # Profondeur maximale de chaque arbre
        learning_rate=0.1,  # Taux d'apprentissage
        random_state=42     # Seed pour la reproductibilité
    )
    
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_xgb)
    cl_rep = classification_report(y_test, y_pred_xgb)
    cm = confusion_matrix(y_test, y_pred_xgb)
    
    print("Accuracy:", accuracy)
    print(cl_rep)
    print("\nConfusion Matrix:")
    print(cm)

    return (xgb, accuracy, cl_rep, cm)

def random_forest_gridsearch(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de classification Random Forest à l'aide d'un GridSearch et évalue ses performances.

    Parameters:
    ----------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).

    Returns:
    -------
    model : RandomForest Classifier
        Le meilleur modèle de classification RandomForest entraîné à l'aide d'un gridsearch.
    
    Best params : dictionnary 
        Les meilleurs paramètres retenus pour l'entraînement du meilleur modèle.

    accuracy : float
        La précision du modèle sur les données de test.

    report : str
        Le rapport de classification détaillant les performances du modèle.

    cm : array
        La matrice de confusion du modèle sur les données de test.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    
    best_rf = grid_search.best_estimator_
    y_pred_RFgs = best_rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_RFgs)
    cl_rep = classification_report(y_test, y_pred_RFgs)
    cm = confusion_matrix(y_test, y_pred_RFgs)
    
    print("Accuracy:", accuracy)
    print(cl_rep)
    print("\nConfusion Matrix:")
    print(cm)
    return (best_rf, grid_search.best_estimator_, accuracy, cl_rep, cm)
    

def xgb_class_gridsearch(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de classification XGB à l'aide d'un Gridsearch et évalue ses performances.

    Parameters:
    ----------
    X_train : array-like or pandas DataFrame
        Les données d'entraînement, de forme (n_samples, n_features).

    y_train : array-like
        Les étiquettes des données d'entraînement, de forme (n_samples,).

    X_test : array-like or pandas DataFrame
        Les données de test, de forme (n_samples, n_features).

    y_test : array-like
        Les étiquettes des données de test, de forme (n_samples,).

    Returns:
    -------
    model : XGB Classifier
        Le meilleur modèle de classification RandomForest entraîné à l'aide d'un gridsearch.
    
    Best params : dictionnary 
        Les meilleurs paramètres retenus pour l'entraînement du meilleur modèle.

    accuracy : float
        La précision du modèle sur les données de test.

    report : str
        Le rapport de classification détaillant les performances du modèle.

    cm : array
        La matrice de confusion du modèle sur les données de test.
    """

    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    
    xgb = XGBClassifier(random_state=42)
    
    #  grille de paramètres à tester
    param_grid = {
        'n_estimators': [50, 100, 200],        # Nombre d'arbres
        'max_depth': [3, 4, 5, 6],             # Profondeur maximale des arbres
        'learning_rate': [0.01, 0.1, 0.2],     # Taux d'apprentissage
        'subsample': [0.8, 1.0],               # Fraction de l'échantillon à utiliser pour l'entraînement de chaque arbre
        'colsample_bytree': [0.8, 1.0],        # Fraction des colonnes à utiliser pour l'entraînement de chaque arbre
        'gamma': [0, 0.1, 0.2]                 # Régularisation par complexité de modèle
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                               scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    # Afficher les meilleurs paramètres
    print("Meilleurs paramètres trouvés : ", grid_search.best_params_)
    
    # Entraîner le modèle avec les meilleurs paramètres trouvés & prédire les valeurs de test
    best_xgb = grid_search.best_estimator_
    y_pred_xgb_gs = best_xgb.predict(X_test)
    
    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred_xgb_gs)
    cl_rep = classification_report(y_test, y_pred_xgb_gs)
    cm = confusion_matrix(y_test, y_pred_xgb_gs)
    
    print("Accuracy:", accuracy)
    print(cl_rep)
    print("\nConfusion Matrix:")
    print(cm)

    return (best_xgb, grid_search.best_params_, accuracy, cl_rep, cm)