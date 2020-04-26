import matplotlib.pyplot as plt
import geopandas
from descartes.patch import Polygon
from geopandas import GeoSeries
from shapely.geometry import Polygon, Point


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# for s in world.name:
#     print(s)
Iran = world[(world.name == "Iran")]
Emirates = world[(world.name == "United Arab Emirates")]
Iraq = world[(world.name == "Iraq")]
Arabia = world[(world.name == "Saudi Arabia")]
Kuwait = world[(world.name == "Kuwait")]
Oman = world[(world.name == "Oman")]

contries = [Iraq, Arabia, Emirates, Kuwait, Oman]

back = Iran.plot(color='black', edgecolor='red')
for c in contries:
    back = c.plot(ax=back, color='black', edgecolor='red')

import pandas as pd

df = pd.DataFrame(
    {'City': ['bandar Abbas', 'Dubai', 'Kowait City', 'Fav', 'Ras', 'Masqat'],
     'Country': ['Iran', 'Emirates', 'Kowait', 'Iraq', 'Arabia', 'Oman'],
     'Latitude': [27.11, 25.15, 29.22, 29.58, 27.45, 23.58],
     'Longitude': [56.16, 55.17, 47.58, 48.28, 49.3, 58.35]})

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

gdf.plot(ax=back, color='red')

print(Iraq)

p = Point(50, 35)
# polygon.to_file('polygon.geojson', )
gdf.to_file(filename='iraq.geojson', driver='GeoJSON')
plt.show()

a = geopandas.read_file('iraq.geojson')
print(a.crs)
