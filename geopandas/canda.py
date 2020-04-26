import matplotlib.pyplot as plt
import geopandas
from descartes.patch import Polygon
from geopandas import GeoSeries
from shapely.geometry import Polygon, Point


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# for s in world.name:
#     print(s)
Canada = world[(world.name == "Canada")]

back = Canada.plot(color='black', edgecolor='red')

import pandas as pd

# df = pd.DataFrame(
#     {'City': ['bandar Abbas', 'Dubai', 'Kowait City', 'Fav', 'Ras', 'Masqat'],
#      'Country': ['Iran', 'Emirates', 'Kowait', 'Iraq', 'Arabia', 'Oman'],
#      'Latitude': [27.11, 25.15, 29.22, 29.58, 27.45, 23.58],
#      'Longitude': [56.16, 55.17, 47.58, 48.28, 49.3, 58.35]})
#
# gdf = geopandas.GeoDataFrame(
#     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
#
# gdf.plot(ax=back, color='red')

# print(Iraq)
back.set_xlim((-70, -50))
back.set_ylim(40, 60)
p = Point(50, 35)
# polygon.to_file('polygon.geojson', )
Canada.to_file(filename='canada.geojson', driver='GeoJSON')
plt.show()

geopandas.read_file('canada.geojson')