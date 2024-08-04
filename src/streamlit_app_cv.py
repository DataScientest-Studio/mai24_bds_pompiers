import streamlit as st
from streamlit_folium import st_folium

import pandas as pd
import numpy as np

# Evaludation des modèles
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score # modèle régression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # modèle classification

# Divers
from utils import dataframe_info, racine_projet, save_model, load_model, load_and_display_plot, load_and_display_interactive_plot
from viz import conf_matrix, pca_plot, xgb_plot_importance
from preprocessing import prepross_reg, prepross_class
from regression import regression_lineaire, ridge_model, lasso_model, elasticnet_model, xgb_model, xgb_gridsearch
from classification import knn_class, decision_tree_class, random_forest_class, xgb_class, random_forest_gridsearch, xgb_class_gridsearch
from deep_learning import deep_learning_dense, deep_learning_improved

############################################################################

# Mise en forme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d69494; /* Fond général */
    }
    .stSidebar {
        background-color: #b9b2b2; /* Fond de la barre latérale */
    }
    .stMarkdown {
        color: #36363d; /* Couleur du texte */
    }
    .stButton>button {
        background-color: #984a5a; /* Couleur des boutons */
        color: white; /* Couleur du texte des boutons */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre
st.title("Temps de réponse de la brigade des pompiers de Londres")

# Barre latérale de navigation
st.sidebar.title("Navigation")
menu = ["0 - Contexte et définition",
        "1 - Présentation des données", 
        "2 - Visualisation des données", 
        "3 - Feature engineering", 
        "4 - Modèles de régression", 
        "5 - Modèles de classification",
        "6 - Modèles de deep learning",
        "7 - Optimisation des modèles", 
        "8 - Bilan & opportunités"]
choice = st.sidebar.radio("Aller à :", menu)

############################################################################

if choice == "0 - Contexte et définition":
    st.header("0 - Contexte & identification du sujet")
    
    st.write("""
        L’objectif de ce projet est d’analyser et/ou d’estimer le *temps de réponse* de la Brigade des Pompiers de Londres.
        \n La brigade des pompiers de Londres est le service d'incendie et de sauvetage **le plus actif du Royaume-Uni et l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde**. L'enjeu d'une organisation efficiente est crucial pour sauver les vies de la plus grande metropole d'Europe de Occidentale.
        """)

    # Insère l'image dans la colonne du milieu
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_path = racine_projet()+'/reports/figures/logo_LFB.png'
        st.image(image_path, caption='Logo de la London Fire Brigade', use_column_width=False , width=300)

    st.write("""
        **Notre objectif principal :**
        - Prédire le temps de mobilisation et de trajet suite à appel du numéro d’urgence (999) et mobilisation d’une brigade.

        **Objectifs secondaires :**
        - Identification des principales données d’entrée influençant les durées d’intervention dans une optique d’amélioration du service : temps de réponse.
        - Réduction du temps d’intervention (souvent liée au temps de réponse) et par conséquent des coûts d’une intervention :
            1) Coût humain (vies sauvées)
            2) Coût matériel (dégâts de l’incendie, ...)
            3) Coût financier (assurances, coût de l’intervention, ...)
        """)

############################################################################

elif choice == "1 - Présentation des données":
    st.header("1 - Présentation des données")

    st.write(
        """
        ### Volumétrie et Description des données
        """
            )
    # Définir les données
    data = {
        "Jeux de données": ["Incidents", "Mobilisation"],
        "Nombre de jeux de données": [2, 3],
        "Période": ["1er janvier 2009 à 2024", "1er janvier 2009 à 2024"],
        "Nombre d'entrées": [1_701_647, 2_373_348],
        "Nombre de variables": [39, 22]
    }
    
    volumetrie = pd.DataFrame(data,)
    st.dataframe(volumetrie)
    
    st.write(
        """   
        [Incidents Dataset](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)  
        [Mobilisations Dataset](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)
    
        **Les données disponibles incluent :**
        - **Données temporelles :** Date, heures, minutes et secondes précises des moments clés liés à l'intervention des pompiers (appel, mobilisation, arrivée sur place, départ, etc.
        - **Données sur les Incidents :** Typologie, nombre d'appels, etc.
        - **Données Géographiques :** position des casernes de pompiers, position de l'incident identifiable à l'aide des numéros de rues, quartiers, etc.
        - **Données de Ressources :** Informations sur les véhicules, équipements, personnels disponibles, nombre de camions mobilisés, etc.
        """
    )
    
    st.write(
        """
        ### Pertinence des données de base
    
        Afin de commencer l'exploration des données, nous avons identifié les variables les plus pertinentes :
    
        - **Date et heure de l'incident :** Comprend l'année, le mois, le jour et l'heure de la mobilisation.
        - **Date et heure d'arrivée sur les lieux :** Permet de calculer le temps nécessaire pour l'arrivée du véhicule.
        - **Lieu de l'incident :** Typologie du lieu (maison, immeuble, bateau, avion, etc.) influençant le matériel et l'expertise nécessaires.
        - **Typologie d'incident :** Intervention classique, incendie, fausse alerte, etc. 
        - **Nombre de véhicules et leur caserne de déploiement :**
            - Station de proximité ou autre station.
            - Localisation de la station.
            - Nombre de véhicules déployés depuis cette station.
        
        Nous souhaitons étudier le temps de réponse, soit la différence entre le "timestamp" de l'appel et celui d'arrivée sur les lieux de l'incident.
        """
    )

    st.write(
        """
        ### Enrichissement du dataset
    
        Pour améliorer et consolider l'entraînement des modèles, des données météo de la ville de Londres ont été intégrées par la suite dans le projet. Elles contiennent à la pour chaque heure les variables suivantes :
        
        - La température, la catégorisation de la météo.
        - L'humidité de l'air, la pluviométrie.
        - La vitesse moyenne et les bourasques du vent.
        """
    )

############################################################################

elif choice == "2 - Visualisation des données":
    st.header("2 - Analyse & visualisation des données")

    sub_tab = st.selectbox("Choix des graphes à afficher :", ["Analyse variable target",
                                                "Analyse variables temporelles",
                                                "Carte des incidents",
                                                "Analyse variables spatiales",
                                                "Analyse des stations",
                                                "Analyse des typologie d'incidents"
                                                ])

    
    if sub_tab == "Analyse variable target":
        load_and_display_plot('target_distribution.png')
        load_and_display_plot('corr_matrix.png')

    elif sub_tab == "Analyse variables temporelles":
        st.write("   => Par années :")
        load_and_display_interactive_plot('distrib_years.html')
        st.write("   => Par mois :")
        load_and_display_interactive_plot('distrib_month.html')
        st.write("   => Par jour :")
        load_and_display_interactive_plot('distrib_day.html')
        st.write("   => Par heure :")
        load_and_display_interactive_plot('distrib_hours.html')

    elif sub_tab == "Carte des incidents":
        st.write("   Temps moyen de <ResponseDuration> groupé par code de District :")
        load_and_display_interactive_plot('mapmeantime.html')

    elif sub_tab == "Analyse variables spatiales":
        load_and_display_plot('distrib_easting.png')
        load_and_display_plot('distrib_northing.png')

    elif sub_tab == "Analyse des stations":    
        load_and_display_interactive_plot('scatter_plot_stations.html')

    elif sub_tab == "Analyse des typologie d'incidents":
        load_and_display_plot('incidenttype_response_duration.png')

############################################################################

elif choice == "3 - Feature engineering":
    st.header("3 - Feature engineering")
    st.write("""
        #### A - Suppression de variables pour causes diverses 
        
        - **Variables redondantes** (par colinéarité ou par codification) - exemple : 'DeployedFromStation_Code', 'PlusCode_Code', 'DelayCodeId', ...
        - **Variables administratives** - 'IncidentNumber', 'Resource_Code'
        - **Variables de sortie** - notamment les différentes variables 'Attendance_Time' 
        - **Variables avec % de NaNs trop élevé** - 'DelayCode', 'PosctCode_full' ...

        #### B - Ajout/traitement de variables 
        
        - **Variables météo :** rain, temperature, humidité, vitesse du vent, ...
        - **Cyclicité des variables de temps :** heure, jour, mois
        - **Compilation de plusieurs variables en une seule** : 'IncidentType' regroupant 'IncidentGroup', 'StopCodeDescription' et 'SpecialServiceType' 

        #### C - Suppression des NaNs restantes

        - Suppression des entrées contenant encore certaines NaN, soit environ
        2500 lignes.
        
        Obtention avant preprocessing d'un **dataset** contenant **25 variables d'entrée** et la target **ResponseDuration** ci-dessous :

    """)

    st.dataframe(pd.read_csv(racine_projet()+'/data/processed/ML_data_info.csv'),width = 900,height=950)


############################################################################

elif choice == "4 - Modèles de régression":
    st.header("4 - Modèles de régression")
    st.write("""
    Preprocessing & affichage des informations du X_train :
    """)
    @st.cache_data
    def preprocessing_reg():
        X_train, y_train, X_test, y_test = prepross_reg(test_size = 0.2, scaler_type='standard', dim_reduction_method='embedded')
        return (X_train, y_train, X_test, y_test)
    
    X_train_reg, y_train_reg, X_test_reg, y_test_reg = preprocessing_reg()
    st.dataframe(dataframe_info(pd.DataFrame(X_train_reg)))

    st.write("""
        **Modèles de Régression Linéaire**
        """)
    
    choix = ['Régression linéaire', 'Ridge','Lasso', 'Elastic Net','XGB Regressor', 'Gridsearch XGB Regressor']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    if option == 'Régression linéaire':
        regressor = load_model('lr_model.pkl')
    elif option == 'Ridge':
        regressor = load_model('ridge_model.pkl')
    elif option == 'Lasso':
        regressor = load_model('lasso_model.pkl')
    elif option == 'Elastic Net':
        regressor = load_model('elasticnet_model.pkl')
    elif option == 'XGB Regressor':
        regressor = load_model('xgb_regressor_model.pkl')
    elif option == 'Gridsearch XGB Regressor':
        regressor = load_model('gridsearch_xgb_regressor_model.pkl')

    y_pred_reg = regressor.predict(X_test_reg)

    df_metrics = pd.DataFrame({'Metric': ['R2', 'RMSE', 'MAE'],
                               'Value': [r2_score(y_test_reg, y_pred_reg), root_mean_squared_error(y_test_reg, y_pred_reg), mean_absolute_error(y_test_reg, y_pred_reg)]})

    st.write("**Résumé des métriques :**")
    st.dataframe(df_metrics)

    st.write("""
    **Présentation des meilleurs résultats obtenus pour chaque modèle**
        """) 
    load_and_display_plot('comparaison_R2_linear.png')

############################################################################

elif choice == "5 - Modèles de classification":
    st.header("5 - Modèles de classification")
    sub_tab = st.selectbox("Affichage de :", ["Catégorisation de la variable cible","Entraînement et résultats"])

    if sub_tab == "Catégorisation de la variable cible":
        st.write("""
        Découpage du temps de réponse en plusieurs catégories pour transformer le problème de régression en problème de classification :
        - Catégorisation du temps d’intervention en classes : “1 - rapide”, “2 - moyen”, “3 - lent”, “4 - très lent”.
            - **Découpage arbitraire de la target**
        """)
        load_and_display_plot('cat_distrib_arb.png', width = 600)
        st.write("""
            - **Découpage selon les quartiles de la distribution**
        """)
        load_and_display_plot('cat_distrib_norm.png', width = 600)
        

    if sub_tab == "Entraînement et résultats":
        
        st.write("""Affichage des informations du X_train et de ses **72 features** après pre-processing :""")
        @st.cache_data
        def preprocessing_class():
            X_train, y_train, X_test, y_test = prepross_class(test_size = 0.2, sampling = 50000, scaler_type='standard', dim_reduction_method='none')
            return (X_train, y_train, X_test, y_test)
        
        X_train_cl, y_train_cl, X_test_cl, y_test_cl = preprocessing_class()
    
        st.dataframe(dataframe_info(pd.DataFrame(X_train_cl)))
    
        st.write("""
            **Modèles de Classification**
            """)
        
        choix = ['KNN', 'Decision Tree', 'Random Forest','Gridsearch RandomForest', 'XGB Classifier', 'Gridsearch XGB Classifier']
        option = st.selectbox('Choix du modèle', choix)
        st.write('Le modèle choisi est :', option)
    
        if option == 'KNN':
            clf = load_model('knn_model.pkl')
        elif option == 'Decision Tree':
            clf = load_model('decisiontree_model.pkl')
        elif option == 'Random Forest':
            clf = load_model('randomforest_model.pkl')
        elif option == 'Gridsearch RandomForest':
            clf = load_model('gridsearch_randomforest_model.pkl')
        elif option == 'XGB Classifier':
            clf = load_model('xgb_classifier_model.pkl')
            st.write(""" **Interprétation du modèle XGB Classifier** """) 
            load_and_display_plot('interpretation.png')
        elif option == 'Gridsearch XGB Classifier':
            clf = load_model('gridsearch_xgb_classifier_model.pkl')
    
        y_pred_cl = clf.predict(X_test_cl)   
        score = clf.score(X_test_cl, y_test_cl)
    
        st.write("**Résumé des métriques :**")
        st.write(f'**Accuracy** : {score}')
        st.write(" ")
            
        st.write("**Matrice de confusion :**")
        st.pyplot(conf_matrix(y_test_cl, y_pred_cl))
    
        st.write("""
            **Présentation des meilleurs résultats obtenus pour chaque modèle**
                """) 
        load_and_display_plot('comparaison_score_class.png')

############################################################################

elif choice == "6 - Modèles de deep learning":
    st.header("6 - Modèles de deep learning")

    @st.cache_data
    def preprocessing_class():
        X_train, y_train, X_test, y_test = prepross_class(test_size = 0.2, sampling = 50000, scaler_type='standard', dim_reduction_method='none')
        return (X_train, y_train, X_test, y_test)
    
    X_train_dl, y_train_dl, X_test_dl, y_test_dl = preprocessing_class()
    y_train_dl = y_train_dl.astype(int)
    y_test_dl = y_test_dl.astype(int)
    
    st.write("""
        **Modèle de Deep Learning**
        """)

    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    choix = ['Dense']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    if option == 'Dense':
        model = tf.keras.models.load_model(racine_projet()+'/models/dense_fullyconnected_model.h5')

    y_pred_dl = model.predict(X_test_dl)
    loss, accuracy = model.evaluate(X_test_dl, y_test_dl)
    y_pred_class = y_pred_dl.argmax(axis=1)

         
    df_metrics = pd.DataFrame({'Metric': ['Accuracy', 'Loss'],'Value': [accuracy, loss]})

    st.write("**Résumé des métriques :**")
    st.dataframe(df_metrics)
    st.write(" ")
        
    st.write("**Matrice de confusion :**")
    st.pyplot(conf_matrix(y_test_dl, y_pred_class))
    
    st.write("""
        **Présentation des performances du modèle Dense de Réseau de Neurones**
            """) 
    load_and_display_plot('performances_deeplearning.png')


############################################################################

elif choice == "7 - Optimisation des modèles":
    st.header("7 - Optimisation des modèles")
     
    # Données du tableau
    data = {
        "Description des différentes tentatives réalisées pour améliorer la performance du modèle": [
            "Passage sur un problème de classification",
            "Choix du modèle",
            "Réduction de dimension via la PCA",
            "Réduction de dimension via LDA",
            "Réduction de dimension via Embedded method",
            "Ajout des données météorologiques",
            "Standardisation MinMax",
            "Standardisation Normale",
            "Ré-équilibrage du jeu de données pour la classification",
            "Taille de l’Undersampling",
            "Utilisation de GridSearch",
            "Utilisation du DeepLearning"
        ],
        "Impact Perf": [
            "+", "+", "=", "-", "+", "=", "-","+", "+", "=", "+", "+"
        ],
        "Impact Temps Calcul": [
            "=", "+", "+", "+", "+", "=", "=", "=", "=", "+", "-", "-"
        ]
    }
    
    # Création du DataFrame
    df = pd.DataFrame(data)
    
    # Fonction pour définir les styles CSS
    def style_table(value):
        color = 'black'
        if value == '-':
            color = 'red'
        elif value == '+':
            color = 'green'
        elif value == '=':
            color = 'grey'
        return f'color: {color};'
    
    # Application des styles
    styled_df = df.style.applymap(style_table, subset=['Impact Perf', 'Impact Temps Calcul'])
    
    # Affichage du DataFrame stylisé dans Streamlit
    st.write("**Description des différentes tentatives réalisées pour améliorer la performance du modèle**")
    st.dataframe(styled_df, height = 470)


############################################################################

elif choice == "8 - Bilan & opportunités":
    st.header("8 - Bilan & Perspectives")
   
    st.write("""
        - Rappel de l'objectif : 
            - **Prédire le temps pour un véhicule de pompiers** pour arriver sur les lieux d'un incident après la décision d’intervenir.
        
        #### BILAN
        - **Régression :**
            - Prédiction du temps de réponse en secondes avec une fenêtre d'erreur.
            - Coefficient de détermination (R2): **0.33**
            - Marge d’erreur : **> 100 secondes**
         
        - **Classification :**
            - Catégorisation du temps d’intervention en classes : “rapide”, “moyen”, “lent”, “très lent”.
            - Accuracy : **< 0.5**
        
        - **Réponse aux objectifs :**
            - Modèles sont **non adaptés** aux attentes potentielles du métier et donc à l'objectif principal.
            - La **visualisation des données** apporte cependant des éléments d'analyse (notamment pour les objectifs secondaires).
        
        #### PERSPECTIVES
        - **Revue du Feature Engineering :**
            - Réanalyse des variables existantes, par exemple la typologie d’incidents.
        - **Amélioration de la Classification :**
            - Collaboration avec la London Fire Brigade (LFB) pour définir les durées d’intervention critiques.
        - **Enrichissement du Dataset :**
            - Ajout de données pertinentes telles que la circulation ou les moyens financiers des brigades.
        - **Segmentation du Problème :**
            - Traitement distinct des problèmes “incendie” et “secours à personne”.
        - **Modèles :**
            - Utilisation de puissances de calcul accrues pour des modèles plus complexes.
            - Développement de réseaux de neurones plus sophistiqués.
                    
    """)
    for _ in range(2):
        st.write("")
        
    load_and_display_plot('london.jpeg')
    st.write("*Photo de la 'City' Londres - par KF Juillet 2023*")

