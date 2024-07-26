from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def regression_lineaire(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de regression_lineaire et évalue ses performances.

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
    model : LinearRegression
        Le modèle de régression linéaire entraîné.

    r2 : float
        Le coefficient de détermination (R²) du modèle sur les données de test.

    rmse : float
        La racine carrée de l'erreur quadratique moyenne (RMSE) sur les données de test.

    mae : float
        L'erreur absolue moyenne (MAE) sur les données de test.
    """
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"r^2: {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    return (lr,r2,rmse,mae)

def ridge_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de régression ridge et évalue ses performances.

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
    model : Ridge
        Le modèle de régression ridge entraîné.

    r2 : float
        Le coefficient de détermination (R²) du modèle sur les données de test.

    rmse : float
        La racine carrée de l'erreur quadratique moyenne (RMSE) sur les données de test.

    mae : float
        L'erreur absolue moyenne (MAE) sur les données de test.
    """
    from sklearn.linear_model import Ridge

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    r2_ridge = r2_score(y_test, y_pred_ridge)
    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    
    print(f"Ridge r^2: {r2_ridge}")
    print(f"Ridge Root Mean Squared Error (RMSE): {rmse_ridge}")
    print(f"Ridge Mean Absolute Error (MAE): {mae_ridge}")
    return (ridge,r2_ridge,rmse_ridge,mae_ridge)


def lasso_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de regression lasso et évalue ses performances.

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
    model : lasso
        Le modèle de regression lasso entraîné.

    r2 : float
        Le coefficient de détermination (R²) du modèle sur les données de test.

    rmse : float
        La racine carrée de l'erreur quadratique moyenne (RMSE) sur les données de test.

    mae : float
        L'erreur absolue moyenne (MAE) sur les données de test.
    """
    from sklearn.linear_model import Lasso

    lasso = Lasso()
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    
    r2_lasso = r2_score(y_test, y_pred_lasso)
    rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    
    print(f"Lasso r^2: {r2_lasso}")
    print(f"Lasso Root Mean Squared Error (RMSE): {rmse_lasso}")
    print(f"Lasso Mean Absolute Error (MAE): {mae_lasso}")
    return (lasso,r2_lasso,rmse_lasso,mae_lasso)

def elasticnet_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle ElasticNet et évalue ses performances.

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
    model : ElasticNet
        Le modèle de régression ElasticNet entraîné.

    r2 : float
        Le coefficient de détermination (R²) du modèle sur les données de test.

    rmse : float
        La racine carrée de l'erreur quadratique moyenne (RMSE) sur les données de test.

    mae : float
        L'erreur absolue moyenne (MAE) sur les données de test.
    """
    from sklearn.linear_model import ElasticNet

    elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X_train, y_train)

    y_pred_elastic = elastic_net.predict(X_test)

    r2_en = r2_score(y_test, y_pred_elastic)
    rmse_en = root_mean_squared_error(y_test, y_pred_elastic)
    mae_en = mean_absolute_error(y_test, y_pred_elastic)

    print(f"r^2: {r2_en}")
    print(f"Root Mean Squared Error (RMSE): {rmse_en}")
    print(f"Mean Absolute Error (MAE): {mae_en}")
    return (elastic_net,r2_en,rmse_en,mae_en)

def xgb_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle XGB Regressor et évalue ses performances.

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
    model : XGB Regressor 
        Le modèle de régression XGB Regressor entraîné.

    r2 : float
        Le coefficient de détermination (R²) du modèle sur les données de test.

    rmse : float
        La racine carrée de l'erreur quadratique moyenne (RMSE) sur les données de test.

    mae : float
        L'erreur absolue moyenne (MAE) sur les données de test.
    """
    from xgboost import XGBRegressor
    
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    r2_xgb = r2_score(y_test, y_pred)
    rmse_xgb = root_mean_squared_error(y_test, y_pred)
    mae_xgb = mean_absolute_error(y_test, y_pred)

    print(f"r^2: {r2_xgb}")
    print(f"Root Mean Squared Error (RMSE): {rmse_xgb}")
    print(f"Mean Absolute Error (MAE): {mae_xgb}")
    return (xgb,r2_xgb,rmse_xgb,mae_xgb)

def xgb_gridsearch(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle Gridsearch XGB Regressor et évalue ses performances.

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
    model : XGB Regressor 
        Le meilleur modèle de régression XGB Regressor entraîné suite à la recherche gridsearch.

    Best params : dictionnary 
        Les meilleurs paramètres retenus pour l'entraînement du meilleur modèle.

    r2 : float
        Le coefficient de détermination (R²) du modèle sur les données de test.

    rmse : float
        La racine carrée de l'erreur quadratique moyenne (RMSE) sur les données de test.

    mae : float
        L'erreur absolue moyenne (MAE) sur les données de test.
    """
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor

    param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1]
                    }
    xgb_model = XGBRegressor(random_state=42)
    
    grid_search = GridSearchCV(estimator=xgb_model, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='neg_mean_squared_error', 
                               n_jobs=-1, 
                               verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    # Afficahege du meilleur score et des meilleurs paramètres
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    
    # Entraînement avec les meilleurs paramètres
    best_xgb = grid_search.best_estimator_
    best_xgb.fit(X_train, y_train)
    y_pred = best_xgb.predict(X_test)

    # Calcul des metrics
    r2_bestxgb = r2_score(y_test, y_pred)
    rmse_bestxgb = root_mean_squared_error(y_test, y_pred)
    mae_bestxgb = mean_absolute_error(y_test, y_pred)
    
    print(f"r^2: {r2_bestxgb}")
    print(f"Root Mean Squared Error (RMSE): {rmse_bestxgb}")
    print(f"Mean Absolute Error (MAE): {mae_bestxgb}")
    
    return (best_xgb, grid_search.best_params_, r2_bestxgb, rmse_bestxgb, mae_bestxgb)
    