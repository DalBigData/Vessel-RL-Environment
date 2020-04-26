import numpy as np
import matplotlib.pyplot as plt
import geopandas
from descartes.patch import Polygon
from geopandas import GeoSeries
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd
# import matplotlib; matplotlib.use("TkAgg")


if True:  # Montreal
      world = geopandas.read_file('montreal_shape/QuebecLand.shp')
      crs = {'init': 'epsg:4326'}
      world = world.to_crs(crs)
      tmp = world.plot()
      plt.xlim([-74.1, -73.7])
      plt.ylim([45.30, 45.50])
      plt.show()
      exit()
world = geopandas.read_file('halifax_shape/HalifaxBuffer50m2.shp')
crs = {'init': 'epsg:4326'}
print(world.crs)
world = world.to_crs(crs)
print(world.crs)
tmp = world.plot()


p = Point(-63.64, 44.68).buffer(0.003)
polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[p])
polygon.plot(color='g', ax=tmp)

pp = [Point(-63.538, 44.602),
      Point(-63.57, 44.655),
      Point(-63.57, 44.66),
      Point(-63.578, 44.662),
      Point(-63.507, 44.615),
      Point(-63.51, 44.585),
      Point(-63.53, 44.59),
      Point(-63.535, 44.585),
      Point(-63.55, 44.61),
      Point(-63.56, 44.615),
      Point(-63.55, 44.62),
      Point(-63.54, 44.634),
      Point(-63.555, 44.635),
      Point(-63.56, 44.65),
      Point(-63.63, 44.68),
      Point(-63.64, 44.7),
      Point(-63.66, 44.71),
      Point(-63.65, 44.69),
      Point(-63.625, 44.695),
      Point(-63.635, 44.687),
      Point(-63.623, 44.687),
      Point(-63.655, 44.703),
      Point(-63.493, 44.59),##
      Point(-63.543, 44.612),
      Point(-63.547, 44.603),
      Point(-63.555, 44.625),
      Point(-63.545, 44.593),

      ]
ss = [0.001 for p in pp]
for ppp, sss in zip(pp, ss):
    p = ppp.buffer(sss)
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[p])
    polygon.plot(color='b', ax=tmp)

p = Point(-63.54, 44.59).buffer(0.001)
polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[p])
polygon.plot(color='tomato', ax=tmp,alpha=0.75)
plt.xticks(np.arange(-63.7, -63.5, 0.01),rotation=45)
plt.yticks(np.arange(44.58, 44.73, 0.01))
# ll = LineString([[-63.555, 44.625], [-63.545, 44.593]]).buffer(0.1)
#
# polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[ll])
# polygon.plot(color='tomato', ax=tmp, alpha=0.75)

plt.grid(True)
plt.show()
