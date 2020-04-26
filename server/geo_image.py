
import server.setting as setting
from server.math.geom_2d import *
from shapely.geometry import Polygon, Point
import geopandas as gpd
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use("TkAgg")
from os import path


class GeoImage:
    def __init__(self, geo_path="", image_path=""):
        self.geo_path = setting.map_path
        if geo_path != "":
            self.geo_path = geo_path
        self.image_path = setting.geo_image_path
        if image_path != "":
            self.image_path = image_path
        self.geo_xlim = setting.x_lim
        self.geo_ylim = setting.y_lim

        self.geo: Polygon = gpd.read_file(self.geo_path).to_crs(setting.map_crs)
        self.resolution = [setting.geo_image_resolution_x, setting.geo_image_resolution_y]
        self.image = [[0 for _ in range(self.resolution[1] + 1)] for _ in range(self.resolution[0] + 1)]
        if setting.geo_image_path != '' and path.exists(self.image_path):
            file = open(self.image_path, 'r')
            lines = file.readlines()
            self.image = eval(lines[0])
        else:
            geo_xdif = self.geo_xlim[1] - self.geo_xlim[0]
            geo_ydif = self.geo_ylim[1] - self.geo_ylim[0]
            geo_xstep = geo_xdif / self.resolution[0]
            geo_ystep = geo_ydif / self.resolution[1]
            x = self.geo_xlim[0]
            while x < self.geo_xlim[1]:
                print('make geo image', round((x - self.geo_xlim[0]) / (self.geo_xlim[1] - self.geo_xlim[0]) * 100, 1))
                y = self.geo_ylim[0]
                while y < self.geo_ylim[1]:
                    xx, yy = self.discrete(x, y)
                    p = Point(x, y)
                    if any(self.geo.contains(p)):
                        self.image[xx][yy] = 1
                    y += geo_ystep
                x += geo_xstep

        lx = []
        ly = []
        sx = []
        sy = []
        for xx in range(self.resolution[0]):
            for yy in range(self.resolution[1]):
                if self.image[xx][yy] == 1:
                    lx.append(xx)
                    ly.append(yy)
                else:
                    sx.append(xx)
                    sy.append(yy)
        # plt.plot(lx, ly, 'r.', sx, sy, 'b.', markersize=10)
        # plt.show()

    def continuous(self, xx, yy):
        geo_xdif = self.geo_xlim[1] - self.geo_xlim[0]
        geo_ydif = self.geo_ylim[1] - self.geo_ylim[0]
        x = (xx / self.resolution[0]) * geo_xdif + self.geo_xlim[0]
        y = (yy / self.resolution[0]) * geo_ydif + self.geo_ylim[0]
        return x, y

    def discrete(self, x, y):
        geo_xdif = self.geo_xlim[1] - self.geo_xlim[0]
        geo_ydif = self.geo_ylim[1] - self.geo_ylim[0]
        xx = round((x - self.geo_xlim[0]) / geo_xdif * self.resolution[0])
        yy = round((y - self.geo_ylim[0]) / geo_ydif * self.resolution[1])
        return xx, yy

    def in_map(self, x, y):
        geo_xdif = self.geo_xlim[1] - self.geo_xlim[0]
        geo_ydif = self.geo_ylim[1] - self.geo_ylim[0]
        xx = round((x - self.geo_xlim[0]) / geo_xdif * self.resolution[0])
        yy = round((y - self.geo_ylim[0]) / geo_ydif * self.resolution[1])
        if xx >= self.resolution[0] or yy >= self.resolution[1]:
            return False
        if xx < 0 or yy < 0:
            return False
        return True

    def in_sea(self, x, y):
        if not self.in_map(x, y):
            return False
        geo_xdif = self.geo_xlim[1] - self.geo_xlim[0]
        geo_ydif = self.geo_ylim[1] - self.geo_ylim[0]
        xx = round((x - self.geo_xlim[0]) / geo_xdif * self.resolution[0])
        yy = round((y - self.geo_ylim[0]) / geo_ydif * self.resolution[1])
        if self.image[xx][yy] == 1:
            return False
        return True

    def in_sea_xxyy(self, xx, yy):
        if xx >= self.resolution[0] or yy >= self.resolution[1]:
            return False
        if xx < 0 or yy < 0:
            return False
        if self.image[xx][yy] == 1:
            return False
        return True

    def get_points_in_radius(self, xx, yy, r):
        res = []
        for xxx in range(xx - r, xx + r + 1):
            res.append([xxx, yy - r])
            res.append([xxx, yy + r])
        for yyy in range(yy - r + 1, yy + r):
            res.append([xx + r, yyy])
            res.append([xx - r, yyy])
        return res

    def get_nearest_field(self, x, y):
        xx, yy = self.discrete(x, y)
        base: Vector2D = Vector2D(x, y)
        for r in range(min(self.resolution)):
            res = self.get_points_in_radius(xx, yy, r)
            find = False
            min_dist = max(self.resolution)
            nearest_point = None
            for p in res:
                if p[0] >= len(self.image) or p[1] >= len(self.image[p[0]]) or self.image[p[0]][p[1]] == 1:
                    find = True
                    rx, ry = self.continuous(p[0], p[1])
                    rp = Vector2D(rx, ry)
                    # print(p, rx, ry)
                    dist = rp.dist(base)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_point = rp
            if find:
                return min_dist
        return 100


if __name__ == "__main__":
    #
    x_lim = setting.x_lim  # Min and max of field x
    y_lim = setting.y_lim

    import random
    import time
    a = time.time()

    geo_path = '../' + setting.map_path
    image_path = '../' + setting.geo_image_path
    gi = GeoImage(geo_path, image_path)

    file = open('../geopandas/montreal_shape/MontrealBuffer50m2_2000t1000round.image', 'w')
    file.write(str(gi.image))
    r = gi.get_points_in_radius(50, 50, 2)
    gi.get_nearest_field(-63.64, 44.7)
    b = time.time()
    # for i in range(1000):
    #     x = random.random() * (xlim[1] - xlim[0]) + xlim[0]
    #     y = random.random() * (ylim[1] - ylim[0]) + ylim[0]
    #     p = Point(x,y)
    #     gi.geo.contains(p)
    # c = time.time()
    # for i in range(1000):
    #     x = random.random() * (xlim[1] - xlim[0]) + xlim[0]
    #     y = random.random() * (ylim[1] - ylim[0]) + ylim[0]
    #     gi.in_sea(x, y)
    # d = time.time()
    #
    # print(b - a, c - b, d - c)