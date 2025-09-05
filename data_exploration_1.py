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
# %%
second_directory = r'C:\Users\ghirg\OneDrive\Visual_studio\My_Packages'
sys.path.append(os.path.abspath(second_directory))

from Harveststats1 import* 
#"C:\Users\ghirg\OneDrive\Visual_studio\My_Packages\Harveststats1.py"
# %%


# Função auxiliar para descobrir zona UTM correta de um ponto
def get_utm_zone(lon, lat):
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'south' if lat < 0 else 'north'
    return zone, hemisphere

# Configurações
SIRGAS_BASE_CODE = 31982  # EPSG base (Zona 20S)

#%%
#RESOLVER DEPOIS!!!!
#expected_field_shapefile = "talhoes.shp"  # Shapefile dos campos

# Carregar dados operacionais (WGS ou UTM desconhecido)
path_fieldview = "C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\FieldView"
shp_fieldview = glob.glob(os.path.join(fr"{path_fieldview}", "*.shp"))
fieldview_gdfs = [gpd.read_file(shp) for shp in shp_fieldview]
combined_fieldview = gpd.GeoDataFrame(pd.concat(fieldview_gdfs, ignore_index = True))

combined_fieldview.to_parquet(fr"{path_fieldview}\\Combinado.parquet")

#%%
for root, dirs, files in os.walk(path_fieldview):
    for file in files:
        if file.lower().endswith(".shp"):
            print(os.path.join(root, file))


#%%

file = r"C:\Users\ghirg\OneDrive\Dados de máquinas\2025\Pontos_colheitas\FieldView\P2.shp"
gdf = gpd.read_file(file)

print("Rows:", len(gdf))
print("Geometry type:", gdf.geom_type.unique())
print("CRS:", gdf.crs)
print("Head:\n", gdf.head())
#%%
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



#%%

#gdf72 = gpd.read_file(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\SIRGAS_7250.shp")

gdf72 = gpd.read_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\7250SIRGASGEO.parquet")
gdf26 = gpd.read_file(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\SIRGAS_2688.shp")

# Verificar SRC original
print("SRC original:", gdf72.crs)
print("SRC original:", gdf26.crs)

#%%

#gdf72.to_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\7250SIRGASGEO.parquet")
gdf26.to_parquet(fr"C:\\Users\\ghirg\\OneDrive\\Dados de máquinas\\2025\\Pontos_colheitas\\2688SIRGASGEO.parquet")


combined_fieldview

#%%
fig, ax = plt.subplots(figsize=(10,10))

gdf26.plot(ax = ax)
gdf72.plot(ax = ax)
combined_fieldview.plot(ax = ax)

#%%

sample_ratio = 0.05  # 1% = ~5k points
gdf26_sample = gdf26.sample(frac=sample_ratio)
gdf72_sample = gdf72.sample(frac=sample_ratio)
combined_sample = combined_fieldview.sample(frac=sample_ratio)

# Now plot the samples
m = gdf26_sample.explore(color="blue", name="GDF26")
gdf72_sample.explore(m=m, color="green", name="GDF72")
combined_sample.explore(m=m, color="red", name="FieldView")

folium.LayerControl().add_to(m)
m.save("sampled_map.html")

#m = folium.Map(location=[-15, -50], zoom_start=5)

m


#%%
m = gdf26[['geometry']].explore(color="blue", name="GDF 26")

# Add other GeoDataFrames using the `m` map object
gdf72[['geometry']].explore(m=m, color="green", name="GDF 72")
combined_fieldview[['geometry']].explore(m=m, color="red", name="FieldView Combined")

# Add a layer control so you can toggle layers

folium.LayerControl().add_to(m)

# Show map (in Jupyter) or save it
m

#%%
print(gdf26.columns, gdf72.columns, combined_fieldview.columns)

#%%
list(gdf26['Dataset'].unique())
#%%
list(gdf72['Dataset'].unique())
#%%
list(combined_fieldview['Dataset'].unique())

#%%

dados = []
fig, axs = plt.subplots(1,1)
axs.boxplot([gdf26['Yld_Mass_D'],gdf72['Yld_Mass_D'], combined_fieldview['Yld_Mass_D'].dropna().values])
print(combined_fieldview['Yld_Mass_D'].dropna().describe())
print(gdf72['Yld_Mass_D'].dropna().describe())
print(gdf26['Yld_Mass_D'].dropna().describe())

#%%
gpd.GeoDataFrame(combined_fieldview[combined_fieldview["Yld_Mass_D"].notna()]).shape

#%%
gpd.GeoDataFrame(combined_fieldview).shape
