import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

lon_point_list = [447123, 448230, 448230, 447123]
lat_point_list = [4948000, 4948000, 4941720, 4941720]


lon_point_list2 = [5, 3, 2]
lat_point_list2 = [1, 3, 1]

polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
polygon_geom2 = Polygon(zip(lon_point_list2, lat_point_list2))

p1 = Point(445000, 4937500).buffer(1)
p2 = Point(460000, 4952500).buffer(1)

print(p1.within(polygon_geom))
print(p2.within(polygon_geom))
print(p1.distance(p2))
print(polygon_geom.contains(p1))
print(polygon_geom.contains(p2))
print(polygon_geom.within(p1))
print(polygon_geom.within(p2))
if polygon_geom2.within(p1):
    print(p1)
if polygon_geom2.within(p2):
    print(p2)


crs = {'init': 'epsg:4326'}
polygon = gpd.GeoDataFrame(index=[0,1,3], crs=crs, geometry=[p1, p2, polygon_geom])



print(polygon.geometry)

polygon.to_file(filename='polygon.geojson', driver='GeoJSON')
# polygon.to_file(filename='polygon.shp', driver="ESRI Shapefile")
polygon.plot()


plt.show()