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

