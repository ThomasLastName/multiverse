
from quality_of_life.my_base_utils import find_root_dir_of_repo, my_warn
from quality_of_life.my_scipy_utils import extend_to_grid
from bnns.utils import load_coast_coords, slosh_heatmap
from matplotlib import pyplot as plt
# from plotly import graph_objects as go
import numpy as np
import os
from bnns import __path__

data_folder = os.path.join( __path__[0], "data" )
os.chdir(data_folder)

#
# ~~~ Plot coastline
try:
    os.chdir("ne_10m_coastline")
    c = load_coast_coords("ne_10m_coastline.shp")
    coast_x, coast_y = c[:,0], c[:,1]
    plt.scatter(coast_x,coast_y)
    plt.show()
    os.chdir("..")
except FileNotFoundError:
    my_warn(f"In order to plot the coastline, go to https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/ and click the `Download coastline` button. Unzip the folder, and move the unzipped folder called `ne_10m_coastline` into the folder {data_folder}")

#
# ~~~ Get the data
from bnns.data.slosh_70_15_15 import out_np, coords_np, inputs_np
# vector_viz( x=coords_np[:,0], y=coords_np[:,1], z=out_np[0] )
x = coords_np[:,0]
y = coords_np[:,1]
N = 3000
z = out_np[N]

#
# ~~~ Plot a heatmap using interpolation
X,Y,Z = extend_to_grid( x, y, z, res=501, method="linear" )
Z = np.nan_to_num(Z,nan=0.)
Z = Z*(Z>0)
plt.figure(figsize=(9,7))
plt.contourf( X, Y, Z, cmap="viridis" )
plt.colorbar(label="Storm Surge Heights")
try:
    plt.plot( coast_x, coast_y, color="black", linewidth=1, label="Coastline" )
except:
    pass
plt.xlim(X.min(),X.max())
plt.ylim(Y.min(),Y.max())
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heightmap")
plt.legend()
plt.tight_layout()
plt.show()

#
# ~~~ Plot a heatmap as a scatterplot without interpolation
plt.figure(figsize=(9,7))
plt.scatter( x, y, c=z, cmap="viridis" )
plt.colorbar(label="Storm Surge Heights")
try:
    plt.plot( coast_x, coast_y, color="black", linewidth=1, label="Coastline" )
except:
    pass
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heightmap")
plt.legend()
plt.tight_layout()
plt.show()

#
# ~~~ Use plt.imshow (extend the heatmap into a grid compatible with plt.imshow(grid)) THIS DOESN'T ACTUALLY WORK BUT I DON'T REALLY KNOW WHY, NOR DO I THINK IT MATTERS
grid_x = np.linspace( x.min(), x.max(), 350 )   # ~~~ 350 == 1+round( (x.max()-x.min())/0.001 ); 0.001 discovered since `np.unique(np.diff(x))` are all multiples of 0.001
grid_y = np.linspace( y.min(), y.max(), 272 )   # ~~~ 272 == 1+round( (y.max()-y.min())/0.001 ); 0.001 discovered since `np.unique(np.diff(y))` are all multiples of 0.001
image = np.full( (350,272), np.nan )
for height, i, j in zip( z, np.searchsorted(grid_x,x), np.searchsorted(grid_y,y) ):
    image[i,j] = height

plt.imshow( image, cmap="viridis", extent=[x.min(),x.max(),y.min(),y.max()] )
try:
    plt.plot( coast_x, coast_y, color="black", linewidth=1, label="Coastline" )
except:
    pass
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heightmap")
plt.legend()
plt.tight_layout()
plt.show()

#
# ~~~ Test the funciton slosh_heatmap()
slosh_heatmap( out=out_np[N], inp=inputs_np[N] )
