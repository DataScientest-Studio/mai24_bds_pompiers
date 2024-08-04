import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from branca.colormap import LinearColormap

from utils import racine_projet

def target_distribution(df, target):
    mean_value = df[target].mean()

    fig = plt.figure(figsize=(8, 6))
    sns.kdeplot(df[target], fill = True, color='blue')

    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.title(f'Distribution de {target} avec affichage de la valeur médiane')
    plt.xlabel('Temps (secondes)')
    plt.ylabel('Densité')

    return fig

def corr_matrix(df, target):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df_numeric.corr()

    # On remet responseDuration au centre de la matrice de correlation
    variable_centre = target
    columns = [variable_centre] + [col for col in corr_matrix.columns if col != variable_centre]
    corr_matrix = corr_matrix.loc[columns, columns]

    fig = plt.figure(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", annot_kws={"size": 7, "color": "black"})
    plt.title('Matrice de Corrélation')
    return fig

def easting_distrib(df):
    # KDE plot EASTING
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(df['Easting_rounded'], bins = 30, color='red')
    plt.title('Distribution de <Easting_rounded>')
    plt.xlabel('m')
    return fig

def northing_distrib(df):
    # KDE plot NORTHING
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(df['Northing_rounded'], bins = 30, color='red')
    plt.title('Distribution de <Northing_rounded>')
    plt.xlabel('m')
    return fig

def easting_northing_plots(df, target):
    # Définir les bins pour Northing_rounded & eating_rounded
    bins_northing = np.linspace(154000, 202000, 50)
    df['Northing_binned'] = pd.cut(df['Northing_rounded'], bins_northing)
    bins_easting = np.linspace(500000, 565000, 50)
    df['Easting_binned'] = pd.cut(df['Easting_rounded'], bins_easting)

    # Groupby pour calculer la médiane de ResponseDuration par bins de Northing_rounded & Easting_rounded
    grouped_northing = df.groupby('Northing_binned')[target].median().reset_index()
    grouped_easting = df.groupby('Easting_binned')[target].median().reset_index()

    # Préparation des données pour l'affichage
    grouped_northing['Northing_binned'] = grouped_northing['Northing_binned'].astype(str)
    grouped_easting['Easting_binned'] = grouped_easting['Easting_binned'].astype(str)

    # Création des subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Plot pour Northing
    sns.barplot(x='Northing_binned', y= target, data=grouped_northing, ax=axes[0])
    axes[0].set_xlabel('Northing_rounded bins')
    axes[0].set_ylabel(f'Average {target}')
    axes[0].set_title(f'Average {target} by Northing_rounded bins')
    axes[0].set_xticklabels(axes[0].get_xticks(), rotation=45)
    axes[0].set_ylim(200, 700)

    # Arrondir les étiquettes des xticks à la virgule près pour Northing
    xticks = axes[0].get_xticks()
    xticks_labels = [round(tick, 1) for tick in xticks]
    axes[0].set_xticklabels(xticks_labels)

    # Plot pour Easting
    sns.barplot(x='Easting_binned', y=target, data=grouped_easting, ax=axes[1])
    axes[1].set_xlabel('Easting_rounded bins')
    axes[1].set_ylabel(f'Average {target}')
    axes[1].set_title(f'Average {target} by Easting_rounded bins')
    axes[1].set_xticklabels(axes[1].get_xticks(), rotation=45)
    axes[1].set_ylim(200, 700)

    # Arrondir les étiquettes des xticks à la virgule près pour Easting
    xticks = axes[1].get_xticks()
    xticks_labels = [round(tick, 1) for tick in xticks]
    axes[1].set_xticklabels(xticks_labels)

    plt.tight_layout()
    
    return fig

def londonmap(df):
    df_geo = df[['ResponseDuration','DateAndTimeMobilised','Northing_rounded','Easting_rounded','Postcode_district']]
    
    # Importation des postcodes du Royaume Uni (fichier télécharger au lien suivant : https://www.freemaptools.com/download-uk-postcode-lat-lng.htm)
    postcodes = pd.read_csv(racine_projet()+'/data/external/ukpostcodes.csv')
    
    #Conservation de la première partie du postcode uniquement dans Postcode_district
    postcodes['postcode'] = postcodes['postcode'].astype(str)
    postcodes['postcode'] = postcodes['postcode'].apply(lambda x: x.lstrip())
    postcodes['Postcode_district'] = postcodes['postcode'].apply(lambda x: x.split(' ')[0])

    dfmap = df_geo.groupby('Postcode_district')['ResponseDuration'].mean().reset_index()
    map = pd.merge(dfmap, postcodes, left_on = 'Postcode_district', right_on = 'Postcode_district', how = 'inner').reset_index()
    map = map.groupby('Postcode_district').agg({'ResponseDuration': 'mean',       # Compter la moyenne de la durée d'intervention
                                                'latitude': 'mean',               # Calculer la moyenne de 'latitude'
                                                'longitude': 'mean'               # Calculer la moyenne de 'longitude'
                                                }).reset_index().reset_index()
    df = pd.DataFrame(map)

    # Initialize the map
    mymap = folium.Map(location=[51.5, 0], zoom_start=10)

    # Créer une palette de couleurs
    colormap = LinearColormap(['green', 'yellow', 'red'], vmin=df['ResponseDuration'].min(), vmax=df['ResponseDuration'].max())

    # Ajouter des marqueurs à la carte
    for index, row in df.iterrows():
        color = colormap(row['ResponseDuration'])
        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=10,
                            color=color,
                            fill=True,
                            fill_color=color,
                            tooltip=f"Value: {row['ResponseDuration']}"  # Ajouter la valeur comme tooltip
                            ).add_to(mymap)

    # Ajouter la légende à la carte
    colormap.caption = 'Response Duration'
    colormap.add_to(mymap)

    # Save the map as an HTML file (optional)
    # mymap.save(racine_projet()+'/reports/figures/mapmeantime.html')
    return mymap


def conf_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    return fig


def pca_plot(pca):

    fig1 = plt.figure()
    plt.xlim(0,72)
    plt.plot(pca.explained_variance_ratio_);

    fig2 = plt.figure()
    plt.xlim(0,72)
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Part de variance expliquée')
    plt.axhline(y = 0.95, color ='r', linestyle = '--')
    plt.plot(pca.explained_variance_ratio_.cumsum());

    return (fig1, fig2)


def xgb_plot_importance(xgb):
    from xgboost import plot_importance

    fig, ax = plt.subplots(figsize=(10, 8))  # Augmenter la taille du graphique
    plot_importance(xgb, importance_type='gain', ax=ax, max_num_features=15, height=0.8)  # Ajuster la hauteur des barres et limiter le nombre de caractéristiques
    
    # Ajouter un titre et ajuster la taille des étiquettes
    plt.title('Importance des Caractéristiques (Gain)', fontsize=16)
    plt.xlabel('Gain', fontsize=14)
    plt.ylabel('Caractéristiques', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    return fig

