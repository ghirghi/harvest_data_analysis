#Sumário
#leitura pasta fieldview
#leitura dados por colhedeira
#MOSTRAR OS DADOS EM UM MAPA
#AVALIAR A DISTRIBUIÇÃO ESPACIAL DELES PARA SABER COMO PROSSEGUIR
#CRIAÇÃO DA COLUNA DataHora a partir de Dataset
#Mostrar os Datasets de valor único, de cada conjunto de dados
#Início de análise estatística dos datasets, descrição dos dados
#Shape dos dados
#Descrição dos dados, sem nan
#Boxplots datasets
#Normalização zscore com grupo normalize_zscore_with_group_boxplot()
#Função explorar explore()
#normalizar por linear
#chamadas para rodar o normalizar por linear
# geração de histogramas, chamada funções explore
#função plotagem matriz de correlação

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import sklearn
import seaborn as sns
from IPython.display import display
from shapely.geometry import Point, Polygon
import geopandas as gpd
import sys
import os
from scipy.stats import linregress
from scipy import stats
from scipy.interpolate import griddata
import rasterio
from rasterio.plot import show
import tkinter as tk
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import glob

from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
import scipy.stats

from scipy.spatial import cKDTree

import dask.dataframe as dd
import pyproj


import pykrige
from pykrige.ok import OrdinaryKriging
from pykrige import variogram_models
import folium
from folium import GeoJson
from folium import LayerControl

second_directory = r'C:\Users\ghirg\OneDrive\Visual_studio\My_Packages'
sys.path.append(os.path.abspath(second_directory))

from Harveststats1 import* 
#"C:\Users\ghirg\OneDrive\Visual_studio\My_Packages\Harveststats1.py"

"""

# Função auxiliar para descobrir zona UTM correta de um ponto
def get_utm_zone(lon, lat):
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'south' if lat < 0 else 'north'
    return zone, hemisphere

# Configurações
SIRGAS_BASE_CODE = 31982  # EPSG base (Zona 20S)
"""




#leitura pasta fieldview

def readfolder(path = None):


    #Leitura de pasta, correndo todos os arquivos shp e anotando os nomes

    path_fieldview = "C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\FieldView"
    shp_fieldview = glob.glob(os.path.join(fr"{path_fieldview}", "*.shp"))
    fieldview_gdfs = [gpd.read_file(shp) for shp in shp_fieldview]
    combined_fieldview = gpd.GeoDataFrame(pd.concat(fieldview_gdfs, ignore_index = True))

    combined_fieldview.to_parquet(fr"{path_fieldview}\\Combinado.parquet")

    file = r"C:\Users\ghirg\OneDrive\Dados de máquinas\2025\Pontos_colheitas\FieldView\P2.shp"
    gdf = gpd.read_file(file)
    
    for root, dirs, files in os.walk(path_fieldview):
        for file in files:
            if file.lower().endswith(".shp"):
                print(os.path.join(root, file))


    print("Rows:", len(gdf))
    print("Geometry type:", gdf.geom_type.unique())
    print("CRS:", gdf.crs)
    print("Head:\n", gdf.head())
    
    gdfs = []
    failed_files = []

    for shp in shp_fieldview:
        try:
            gdf = gpd.read_file(shp)
            gdf["source_file"] = os.path.basename(shp)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Failed to read {shp}: {e}")
            failed_files.append(shp)

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    combined_gdf["source_file"].unique()
    combined_gdf["source_file"].value_counts()

    return combined_fieldview



combined_fieldview = readfolder()

#leitura dados por colhedeira

def read_harvester():

    #gdf72 = gpd.read_file(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\SIRGAS_7250.shp")

    gdf72 = gpd.read_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\7250SIRGASGEO.parquet")
    gdf26 = gpd.read_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\2688SIRGASGEO.parquet")
    #gdf26 = gpd.read_file(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\SIRGAS_2688.shp")

    # Verificar SRC original
    print("SRC original:", gdf72.crs)
    print("SRC original:", gdf26.crs)



    #gdf72.to_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\7250SIRGASGEO.parquet")
    #gdf26.to_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\2688SIRGASGEO.parquet")

    return gdf72, gdf26

gdf72, gdf26 = read_harvester()



#MOSTRAR OS DADOS EM UM MAPA
#AVALIAR A DISTRIBUIÇÃO ESPACIAL DELES PARA SABER COMO PROSSEGUIR

def showmap(data1, data2, data3):

    sample_ratio = 0.05  # 1% = ~5k points
    data2_sample = data2.sample(frac=sample_ratio)
    data1_sample = data1.sample(frac=sample_ratio)
    combined_sample = combined_fieldview.sample(frac=sample_ratio)

    # Now plot the samples
    m = data2_sample.explore(color="blue", name="GDF26")
    data1_sample.explore(m=m, color="green", name="GDF72")
    combined_sample.explore(m=m, color="red", name="FieldView")

    folium.LayerControl().add_to(m)
    m.save("sampled_map.html")

    #m = folium.Map(location=[-15, -50], zoom_start=5)

    m


    m = data2[['geometry']].explore(color="blue", name="GDF 26")

    # Add other GeoDataFrames using the `m` map object
    data1[['geometry']].explore(m=m, color="green", name="GDF 72")
    data3[['geometry']].explore(m=m, color="red", name="FieldView Combined")

    # Add a layer control so you can toggle layers

    folium.LayerControl().add_to(m)

    # Show map (in Jupyter) or save it
    m



#CRIAÇÃO DA COLUNA DataHora a partir de Dataset


def mostra_dataset(data):
    print(data['Dataset'].unique())


    data['DataHora'] = data['Dataset'].str.extract(r'(^[\d/:-]+)')
    data['DataHora'] = pd.to_datetime(data['DataHora'], format='%d/%m/%y-%H:%M:%S')
    
    

mostra_dataset(gdf72)
mostra_dataset(gdf26)


#Mostrar as colunas dos conjuntos de dados


print(gdf26.columns, gdf72.columns, combined_fieldview.columns)
#Mostrar os Datasets de valor único, de cada conjunto de dados
#Datasets sâo separados por horário e dia
#Datasets possuem identificação por máquina



#Comparação de datas e horários entre datasets
#Objetivo é encontrar datasets iguais entre as máquinas
#Nâo implica que é no mesmo local

for i in gdf72['Dataset'].unique():
    #if 
    for j in gdf26['Dataset'].unique():
        if(i == gdf26['Dataset'].unique()[0]):
            print(i == gdf26['Dataset'].unique()[j])



#Início de análise estatística dos datasets, descrição dos dados



dados = []
fig, axs = plt.subplots(1,1)
axs.boxplot([gdf26['Yld_Mass_D'],gdf72['Yld_Mass_D'], combined_fieldview['Yld_Mass_D']])
print(combined_fieldview['Yld_Mass_D'].dropna().describe())
print(gdf72['Yld_Mass_D'].dropna().describe())
print(gdf26['Yld_Mass_D'].dropna().describe())


fig, axs = plt.subplots(1,1)
axs.boxplot([gdf26['Yld_Mass_D'].dropna().values,gdf72['Yld_Mass_D'].dropna().values, combined_fieldview['Yld_Mass_D'].dropna().values])





#Exploração de dados
#Shape dos dados
def exploredata(data1, data2, data3):
    print(data1.shape,
    data2.shape,
    data3.shape)

exploredata(gdf72, gdf26, combined_fieldview)
exploredata(gdf26['Yld_Mass_D'].dropna().values,gdf72['Yld_Mass_D'].dropna().values, combined_fieldview['Yld_Mass_D'].dropna().values)

#Descrição dos dados, sem nan
def estatisticas(data1, data2, data3, param = None):


    print(data1['Yld_Mass_D'].dropna().describe())
    print(data2['Yld_Mass_D'].dropna().describe())
    print(data3['Yld_Mass_D'].dropna().describe())

    if param != None:
        
        sns.histplot(data1[param])
        sns.histplot(data2[param])
        sns.histplot(data3[param])
    else:

        data1.hist()
        data2.hist()
        data3.hist()

estatisticas(gdf72, gdf26, combined_fieldview, 'Yld_Mass_D')

#Boxplots datasets

def boxplot_dataset(data1, dataparam, param = None):
    datasets = list(data1[dataparam].unique())

    data_to_plot = []
    xlabels = []
    
    for i in datasets: 
        
        subset = data1[data1[dataparam] == i][param].dropna().values
        data_to_plot.append(subset)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.boxplot(data_to_plot)

datasets = list(gdf72['Dataset'].unique())
datasets

boxplot_dataset(gdf72,'Dataset', 'Yld_Mass_D')
boxplot_dataset(gdf26,'Dataset', 'Yld_Mass_D')
boxplot_dataset(combined_fieldview,'Dataset', 'Yld_Mass_D')



#Normalização zscore com grupo normalize_zscore_with_group_boxplot()

def normalize_zscore_with_group_boxplot(data, attribute, group_col=None, ignore_outliers=True, outlier_threshold=3):
    # Always drop NaNs for consistent processing
    original_data = data[attribute].dropna()

    # Detect outliers
    z_scores_all = (original_data - original_data.mean()) / original_data.std()
    outlier_mask = np.abs(z_scores_all) > outlier_threshold

    # Define filtered data (used for mean/std if ignoring outliers)
    if ignore_outliers:
        filtered_data = original_data[~outlier_mask]
    else:
        filtered_data = original_data

    # Calculate normalization parameters
    mean = filtered_data.mean()
    std = filtered_data.std()

    # Apply normalization to all data
    data[f'{attribute}_zscore'] = (data[attribute] - mean) / std

    # Use consistent data for plotting/stats based on `ignore_outliers`
    normalized_data_to_plot = data.loc[~outlier_mask if ignore_outliers else data[attribute].notna(), f'{attribute}_zscore']
    original_data_to_plot = filtered_data  # already excludes or includes outliers

    # Descriptive statistics
    print("Original Data:")
    print(original_data_to_plot.describe())
    print("\nNormalized Data:")
    print(normalized_data_to_plot.describe())

    # Histogram Plot
    plt.figure(figsize=(12, 5))
    sns.histplot(original_data_to_plot, color='blue', label='Original', kde=True)
    sns.histplot(normalized_data_to_plot, color='orange', label='Z-Score Normalized', kde=True)
    plt.title(f'Histogram - {attribute} (ignore_outliers={ignore_outliers})')
    plt.legend()
    
    plt.show()
    

    # Boxplot: Original vs Normalized
    plt.figure(figsize=(8, 5))
    plt.boxplot([original_data_to_plot, normalized_data_to_plot],
                labels=['Original', 'Normalized'])
    plt.title(f'Boxplot - {attribute} (ignore_outliers={ignore_outliers})')
    plt.show()

    # Grouped boxplots if group_col is provided
    if group_col:
        datasets = list(data[group_col].dropna().unique())
        data_to_plot = []
        labels = []

        for group in datasets:
            group_data = data[data[group_col] == group]
            group_values = group_data[attribute].dropna()

            if ignore_outliers:
                group_z = (group_values - mean) / std
                group_z = group_z[np.abs(group_z) <= outlier_threshold]
            else:
                group_z = (group_values - mean) / std

            data_to_plot.append(group_z)
            labels.append(str(group))

        if data_to_plot:
            plt.figure(figsize=(10, 6))
            plt.boxplot(data_to_plot, labels=labels)
            plt.title(f'Normalized Z-Score Boxplot by {group_col}')
            plt.xlabel(group_col)
            plt.ylabel(f'{attribute}_zscore')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    return data


normalize_zscore_with_group_boxplot(combined_fieldview, 'Yld_Mass_D', group_col='Dataset', ignore_outliers=False)





#%%
#Função explorar explore()



def explore(data, attribute, dohist = None, 
            group_col=None, method='max', ignore_outliers=True, outlier_threshold=3,save = False):
    """
    Normalizes a numeric column by group (e.g., by 'Dataset') using percent-based scaling.
    """

    # --- Check dtype: skip non-numerical columns ---
    if not np.issubdtype(data[attribute].dtype, np.number):
        print(f"Skipping non-numeric column: {attribute}")
        return data

    col_norm = f'{attribute} no outliers'
    original_data = data[attribute].dropna()

    # Compute z-scores and identify outliers
    z_scores = (original_data - original_data.mean()) / original_data.std()
    outlier_mask = np.abs(z_scores) > outlier_threshold
    data_no_outliers = original_data[~outlier_mask]

    # Print descriptive stats
    #print(f"\n===== {attribute} =====")
    #print("Original Data (with outliers):")
    #print(original_data.describe())
    #print("\nOriginal Data (without outliers):")
    #print(data_no_outliers.describe())

    if dohist:
        # Histogram
        plt.figure(figsize=(12, 5))
        sns.histplot(original_data, label='Original', color='blue', kde=True)
        sns.histplot(data_no_outliers.dropna(), label='Original (no outiliers)', color='green', kde=True)
        #salvar histogramas
    
        plt.title(f'Histogram - {attribute} vs {col_norm}')
        plt.legend()
        plt.show()

        if save:
            plt.savefig(fr"C:\Users\ghirg\OneDrive\Curso Engenharia\2025S1\IC Mapas colheita\Plots histogramas\{attribute}_hist.png", dpi=300, bbox_inches="tight")
            
    # Boxplot comparison
    plt.figure(figsize=(8, 5))
    #plt.boxplot([original_data, data_no_outliers.dropna()],
     #           labels=['Original (All)', 'Original (No Outliers)'])
    
    plt.boxplot([original_data.dropna()],
                labels=['Original (All)'])
    
    plt.show()
    plt.title(f'Boxplot - {attribute} Variants')

    plt.boxplot([data_no_outliers.dropna()],
                labels=['Original (No Outliers)'])

    plt.title(f'Boxplot - {attribute} Variants')
    plt.show()

    # Grouped boxplot
    if group_col:
        datasets = list(data[group_col].dropna().unique())
        data_to_plot = []
        labels = []

        for group in datasets:
            subset = data[data[group_col] == group][attribute].dropna().values
            data_to_plot.append(subset)
            labels.append(str(group))

        if data_to_plot:
            plt.figure(figsize=(12, 6))
            plt.boxplot(data_to_plot, labels=labels)
            plt.title(f'{attribute} Boxplot by {group_col}')
            plt.xlabel(group_col)
            plt.ylabel(attribute)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    return data

#%%

#normalizar por linear
def normalize_percent_by_group(data, attribute, group_col=None, method='max', ignore_outliers=True, outlier_threshold=3):
    """
    Normalizes a numeric column by group (e.g., by 'Dataset') using percent-based scaling.

    Parameters:
    - data: DataFrame
    - attribute: the name of the column to normalize
    - group_col: the column to group by (e.g., 'Dataset')
    - method: 'max' for max-based scaling, 'minmax' for full min-max scaling
    - ignore_outliers: whether to remove outliers before normalization
    - outlier_threshold: z-score threshold for outlier removal

    Returns:
    - DataFrame with new normalized column
    - Plots histograms and boxplots comparing original and normalized data
    """
    col_norm = f'{attribute}_percent_normalized'
    original_data = data[attribute].dropna()

    # Compute z-scores and identify outliers
    z_scores = (original_data - original_data.mean()) / original_data.std()
    outlier_mask = np.abs(z_scores) > outlier_threshold
    data_no_outliers = original_data[~outlier_mask]

    def normalize_group(x):
        if ignore_outliers:
            z = (x - x.mean()) / x.std()
            x = x[(np.abs(z) <= outlier_threshold)]
        if method == 'max':
            return 100 * x / x.max() if x.max() != 0 else x
        elif method == 'minmax':
            return 100 * (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
        else:
            return x

    if group_col:
        data[col_norm] = data.groupby(group_col)[attribute].transform(normalize_group)
    else:
        data[col_norm] = normalize_group(data[attribute].dropna())

    # Print descriptive stats
    #print("Original Data (with outliers):")
    #print(original_data.describe())
    #print("\nOriginal Data (without outliers):")
    #print(data_no_outliers.describe())
    #print("\nNormalized Data:")
    #print(data[col_norm].dropna().describe())

    # Histogram
    plt.figure(figsize=(12, 5))
    sns.histplot(original_data, label='Original', color='blue', kde=True)
    plt.title(f'Histogram - {attribute} original')
    plt.legend()
    plt.show()

    sns.histplot(data[col_norm].dropna(), label='Normalized', color='green', kde=True)
    plt.title(f'Histogram - {attribute} {col_norm}')
    plt.legend()
    plt.show()

    # Boxplot comparison
    plt.figure(figsize=(8, 5))
    plt.boxplot([original_data, data_no_outliers, data[col_norm].dropna()],
                labels=['Original (All)', 'Original (No Outliers)', 'Normalized'])
    plt.title(f'Boxplot - {attribute} Variants')
    plt.show()

    # Grouped boxplot
    if group_col:
        datasets = list(data[group_col].dropna().unique())
        data_to_plot = []
        labels = []

        for group in datasets:
            subset = data[data[group_col] == group][col_norm].dropna().values
            data_to_plot.append(subset)
            labels.append(str(group))

        if data_to_plot:
            plt.figure(figsize=(12, 6))
            plt.boxplot(data_to_plot, labels=labels)
            plt.title(f'{col_norm} Boxplot by {group_col}')
            plt.xlabel(group_col)
            plt.ylabel(col_norm)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    return data


#chamadas para rodar o normalizar por linear
#%%
norm_prct_fieldview = normalize_percent_by_group(
    data=combined_fieldview,
    attribute='Yld_Mass_D',
    group_col='Dataset',
    method='max',              # or 'minmax'
    ignore_outliers=False,      # or False
    outlier_threshold=3        # standard z-score threshold
)


norm_prct_72 = normalize_percent_by_group(
    data=gdf72,
    attribute='Yld_Mass_D',
    group_col='Dataset',
    method='max',              # or 'minmax'
    ignore_outliers=False,      # or False
    outlier_threshold=3        # standard z-score threshold
)

norm_prct_26 = normalize_percent_by_group(
    data=gdf26,
    attribute='Yld_Mass_D',
    group_col='Dataset',
    method='max',              # or 'minmax'
    ignore_outliers=False,      # or False
    outlier_threshold=3        # standard z-score threshold
)

#%%

norm_prct_72 = normalize_percent_by_group(
    data=gdf72,
    attribute='Yld_Mass_D',
    group_col='Dataset',
    method='max',              # or 'minmax'
    ignore_outliers=True,      # or False
    outlier_threshold=3        # standard z-score threshold
)


#%%
norm_prct_fieldview = normalize_percent_by_group(
    data=combined_fieldview,
    attribute='Yld_Mass_D',
    group_col='Dataset',
    method='max',              # or 'minmax'
    ignore_outliers=False,      # or False
    outlier_threshold=3        # standard z-score threshold
)


#%%
#geração de histogramas, chamada funções explore
maquinas = combined_fieldview.Dataset.str[-10:].unique()

"""
for i in maquinas:
    print(i)
    bolha = combined_fieldview[combined_fieldview['Dataset'].str[-10:] == i]

    explore(
    data=bolha,
    attribute='Yld_Mass_D',
    group_col='Dataset'
)
"""
    

#%%

"""
for i in gdf72.columns:
    print(i)


    explore(
        data = gdf72,
        attribute=i
    )
"""
#%%
#VENHA AQUI
explore(
    data=gdf26,
    attribute='Yld_Mass_D',
    group_col='Dataset'
)
#%%

explore(
    data=gdf72,
    attribute='Yld_Mass_D'
    
)
 
 #%%
#gdf723 = gdf72.drop(columns = ["Yld_Mass_D_percent_normalized", "Distance_m" >= 5, "Distance_m" <= 0.3, "Speed_km_h" >= 15])
gdf723 = gdf72[(gdf72['Distance_m']<=5) & (gdf72['Distance_m'] >=0.3) & (gdf72['Speed_km_h'] <=15)].drop(columns=['Yld_Mass_D_percent_normalized', 'Obj__Id'])




gdf723.describe()
#%%

print(gdf723.columns[:3])

explore(
    data=gdf723,
    attribute=['DataHora'])


#%%

#%%
for i in gdf723.columns:
    print(f'Atributo: {i}\n')
    explore(
    data=gdf723,
    attribute=i
    
)
    
#%%
def boxplots2(data, attribute, outlier_threshold = 3):

    if not np.issubdtype(data[attribute].dtype, np.number):
            print(f"Skipping non-numeric column: {attribute}")
            return data

    col_norm = f'{attribute} no outliers'
    original_data = data[attribute].dropna()

        # Compute z-scores and identify outliers
    z_scores = (original_data - original_data.mean()) / original_data.std()
    outlier_mask = np.abs(z_scores) > outlier_threshold
    data_no_outliers = original_data[~outlier_mask]

 

#%%

#função plotagem matriz de correlação

def plot_correlation_matrix(data, columns=None, method='pearson', title='Correlation Matrix',
                            cmap='coolwarm', annot=True, clean=True, vmin=-1, vmax=1):
    """
    Plots a correlation matrix with automatic cleaning.

    Parameters:
    - data: the input DataFrame
    - columns: optional list of column names to include
    - method: correlation method ('pearson', 'spearman', or 'kendall')
    - title: plot title
    - cmap: color map for heatmap
    - annot: show correlation values on the matrix
    - clean: whether to auto-clean (drop non-numeric, NaNs, constants)
    - vmin, vmax: scale for color range
    """
    # Step 1: Select desired columns
    if columns is not None:
        df_corr = data[columns].copy()
    else:
        df_corr = data.copy()

    if clean:
        df_corr = df_corr.select_dtypes(include='number')  # keep only numeric
        df_corr = df_corr.loc[:, df_corr.std() > 0]          # remove constant columns
        df_corr = df_corr.dropna()                           # remove rows with NaNs



        numeric_columns = combined_fieldview.select_dtypes(include='number').columns
        filtered_data = combined_fieldview[numeric_columns]

        # Remove constant columns
        filtered_data = filtered_data.loc[:, filtered_data.std(numeric_only=True) > 0]

        # Compute correlation with pairwise NaN handling
        corr_matrix = filtered_data.corr(method=method)

    if df_corr.shape[1] < 2:
        print("Not enough valid numeric columns to compute correlation.")
        return


    """
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix with NaN-safe Pairwise Logic")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=annot
                , cmap=cmap, fmt=".2f",
                  linewidths=0.5, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


plot_correlation_matrix(gdf72, clean=True)


#%%

for i in maquinas:
    print(i)
    bolha = combined_fieldview[combined_fieldview['Dataset'].str[-10:] == i]

    #explore(
    #data=bolha,
    #attribute='Yld_Mass_D',
    #group_col='Dataset')

    plot_correlation_matrix(bolha, clean=True)

    print(bolha.columns)

#%%


pd.DataFrame([combined_fieldview[combined_fieldview['Dataset'].str[-10:] == maquinas[0]].columns,
combined_fieldview[combined_fieldview['Dataset'].str[-10:] == maquinas[1]].columns])




#%%

def plot_cross_correlation_matrix(data, x_vars, y_vars, method='pearson', title='Cross-Correlation Matrix',
                                  cmap='coolwarm', annot=True, vmin=-1, vmax=1, figsize=(20, 15),
                                  save_path=None, return_matrix=False):
    """
    Plots a correlation heatmap between two different sets of variables.

    Parameters:
    - data: pd.DataFrame
    - x_vars: list of column names to be on the x-axis (group 2)
    - y_vars: list of column names to be on the y-axis (group 1)
    - method: correlation method ('pearson', 'spearman', 'kendall')
    - title: title of the plot
    - cmap: color map for the heatmap
    - annot: whether to annotate the heatmap
    - vmin, vmax: limits for color scaling
    - figsize: figure size
    - save_path: optional path to save the figure
    - return_matrix: if True, returns the cross-correlation matrix

    Returns:
    - pd.DataFrame (optional): The cross-correlation matrix
    """
    # Step 1: Filter numeric columns
    df = data.copy()
    x_vars = [col for col in x_vars if pd.api.types.is_numeric_dtype(df[col])]
    y_vars = [col for col in y_vars if pd.api.types.is_numeric_dtype(df[col])]

    # Step 2: Create the cross-correlation matrix
    corr_matrix = pd.DataFrame(index=y_vars, columns=x_vars, dtype=float)

    for y in y_vars:
        for x in x_vars:
            corr = df[[x, y]].corr(method=method).iloc[0, 1]  # upper-right correlation
            corr_matrix.loc[y, x] = corr

    # Step 3: Plot the matrix
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm_r', vmin=vmin, vmax=vmax, fmt=".2f", linewidths=1,square = False)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


    if return_matrix:
        return corr_matrix
    


#plot_cross_correlation_matrix(gdf72,x_vars = gdf72.columns[:25], y_vars = gdf72.columns[25:], method='spearman', title="Spearman Correlation")
plot_cross_correlation_matrix(gdf72,x_vars = gdf72.columns, y_vars = gdf72.columns, method='spearman', title="Spearman Correlation")
plot_cross_correlation_matrix(gdf26,x_vars = gdf26.columns, y_vars = gdf26.columns, method='spearman', title="Spearman Correlation")
plot_cross_correlation_matrix(combined_fieldview,x_vars = combined_fieldview.columns, y_vars = combined_fieldview.columns, method='spearman', title="Spearman Correlation")
#plot_cross_correlation_matrix(gdf26, method='spearman', title="Spearman Correlation")
#%%
print(combined_fieldview.columns[:25])
print(combined_fieldview.columns[25:])
print(combined_fieldview.columns)

#%%
print(np.triu)

#%%
plot_cross_correlation_matrix(gdf72,x_vars = gdf72.columns, y_vars = gdf72.columns, method='pearson', title="Pearson Correlation")
plot_cross_correlation_matrix(gdf26,x_vars = gdf26.columns, y_vars = gdf26.columns, method='pearson', title="Pearson Correlation")



# %%
def plot_corr_auto(
    name, data: pd.DataFrame,
    x_vars=None,
    y_vars=None,
    method: str = "pearson",
    triangular: bool = True,      # only used when square (x==y)
    hide_upper: bool = True,      # lower triangle shown if True
    show_diag: bool = True,       # keep the 1.0 diagonal
    cmap: str = "coolwarm_r",
    annot: bool = True,
    vmin: float = -1,
    vmax: float =  1,
    figsize=(20, 20),
    title: str | None = None, save = False
):
    df = data.copy()

    # Auto-pick numeric columns if none provided
    if x_vars is None and y_vars is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_vars = cols
        y_vars = cols
    elif x_vars is None:
        x_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    elif y_vars is None:
        y_vars = df.select_dtypes(include=[np.number]).columns.tolist()

    # Keep only existing numeric columns
    x_vars = [c for c in x_vars if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    y_vars = [c for c in y_vars if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not x_vars or not y_vars:
        raise ValueError("No numeric columns found for x_vars or y_vars.")

    # Build correlation matrix
    if set(x_vars) == set(y_vars) and len(x_vars) == len(y_vars):
        # Square correlation
        corr = df[x_vars].corr(method=method).loc[y_vars, x_vars]
    else:
        # Rectangular cross-correlation
        corr = pd.DataFrame(index=y_vars, columns=x_vars, dtype=float)
        for y in y_vars:
            for x in x_vars:
                corr.loc[y, x] = df[[x, y]].corr(method=method).iloc[0, 1]

    # Optional triangular mask (only meaningful if square)
    mask = None
    is_square = (corr.shape[0] == corr.shape[1]) and (list(corr.index) == list(corr.columns))
    if triangular and is_square:
        mask = np.zeros_like(corr, dtype=bool)
        if hide_upper:
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=0 if show_diag else 1)
        else:
            mask = np.tril(np.ones(corr.shape, dtype=bool), k=0 if show_diag else -1)

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fmt=".2f",
        linewidths=1,
        square=is_square,
    )
    if title is None:
        title = "Correlation (auto)" if is_square else "Cross-correlation (auto)"
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        plt.savefig(fr"C:\Users\ghirg\OneDrive\Curso Engenharia\2025S1\IC Mapas colheita\Plots histogramas\{title}_hist.png", dpi=300, bbox_inches="tight")
    
    plt.show()

    return corr

plot_corr_auto("7250", gdf72.drop(columns = ['Obj__Id','Yld_Mass_D_percent_normalized', 'Dataset', 'Field', 'Eng_Power_']), method = 'pearson', save = True)
plot_corr_auto("7250",gdf72.drop(columns = ['Obj__Id','Yld_Mass_D_percent_normalized', 'Dataset', 'Field', 'Eng_Power_']), method = 'spearman', save = True)

#%%
plot_corr_auto("2688",gdf26.drop(columns = ['Obj__Id','Wind_Speed', 'Wind_Dir', 'Air_Temp__', 'Humidity__','Yld_Mass_D_percent_normalized', 'Dataset', 'Field']), method = 'pearson', save = True)
plot_corr_auto("2688",gdf26.drop(columns = ['Obj__Id','Wind_Speed', 'Wind_Dir', 'Air_Temp__', 'Humidity__','Yld_Mass_D_percent_normalized', 'Dataset', 'Field']), method = 'spearman', save = True)
#%%

plot_corr_auto("FieldView", combined_fieldview, method = 'pearson', save = True)
plot_corr_auto("FieldView",combined_fieldview, method = 'spearman', save = True)

# %%
