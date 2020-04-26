from collections import defaultdict
from heapq import *
from server.setting import x_lim, y_lim, planner_helper_points
import server.setting as setting
from present.color import Color
import matplotlib.pyplot as plt
from server.geo_image import GeoImage
from server.math.geom_2d import *
from shapely.geometry import Point,LineString
import geopandas as gpd
from sys import maxsize
# import matplotlib; matplotlib.use("TkAgg")
from math import inf
from itertools import product
import os.path
from os import path
import numpy as np
import random


class Planner:
    x_res = setting.plan_resolution[0]
    y_res = setting.plan_resolution[1]
    x_dist = (x_lim[1] - x_lim[0]) / x_res
    y_dist = (y_lim[1] - y_lim[0]) / y_res
    rounder = 6

    def __init__(self, use_here=False, just_show_graph=False):
        self.pos2key = {}
        self.key2pos = {}
        self.graph = []
        self.nodes = []
        dir_path = ('../' if use_here else '') + setting.map_path
        geo_path = ('../' if use_here else '') + setting.shape_path
        image_path = ('../' if use_here else '') + setting.geo_image_path
        self.geo_image = GeoImage(geo_path, image_path)

        next_path = dir_path + f'{setting.city_name}_close_path_{int(Planner.x_res)}_{int(Planner.y_res)}.npy'
        if not path.exists(next_path):
            print('is not exist closest path')
            self.make_dict()

            print('dict made')
            pos2key = np.zeros((len(self.pos2key), 3))
            i = 0
            for pos in self.pos2key.keys():
                pos2key[i][0] = pos[0]
                pos2key[i][1] = pos[1]
                pos2key[i][2] = self.pos2key[pos]
                i += 1
            f = open(next_path[:-4] + '_pos2key.npy', 'wb')
            np.save(f, pos2key)
            f.close()
            print('start to make graph')
            self.make_graph()
            self.show_graph()
            if just_show_graph:
                exit()
            print(self.graph)
            print('graph made')
            self.floyd_warshall(len(self.pos2key.keys()), self.graph)
            self.nxt = np.array(self.nxt)
            f = open(next_path, 'wb')
            np.save(f, self.nxt)
            f.close()
        else:
            print('exist closest')
            f = open(next_path, 'rb')
            self.nxt = np.load(f)
            f.close()

            f = open(next_path[:-4] + '_pos2key.npy', 'rb')
            pos2key = np.load(f)
            f.close()
            self.pos2key = {}
            self.key2pos = {}

            for i in range(pos2key.shape[0]):
                self.pos2key[(pos2key[i][0], pos2key[i][1])] = int(pos2key[i][2])
                self.key2pos[int(pos2key[i][2])] = (pos2key[i][0], pos2key[i][1])

            print('halifax close was read')

    def make_dict(self):
        number = 1
        x = x_lim[0]
        while x <= x_lim[1]:
            y = y_lim[0]
            while y <= y_lim[1]:
                xx = round(x, Planner.rounder)
                yy = round(y, Planner.rounder)
                if not any(self.geo_image.geo.contains(Point(xx, yy))):
                    self.pos2key[(xx, yy)] = number
                    number += 1
                y += Planner.y_dist
            x += Planner.x_dist

        for p in planner_helper_points:
            xx = round(p[0], Planner.rounder)
            yy = round(p[1], Planner.rounder)
            if not any(self.geo_image.geo.contains(Point(xx, yy))):
                self.pos2key[(xx, yy)] = number
                number += 1

        for pos in self.pos2key.keys():
            self.key2pos[self.pos2key[pos]] = pos

    def make_graph(self):
        counter = 0
        counter_size = len(self.pos2key.keys())
        for xy in self.pos2key.keys():
            counter += 1
            print('make graph:', counter / counter_size * 100)
            number = self.pos2key[xy]
            x = xy[0]
            y = xy[1]
            res = Planner.get_nears(x, y)
            bad_n = []
            for n in res:
                newres = Planner.get_nears(n[0], n[1])
                has_bad_n = False
                for nn in newres:
                    if not self.geo_image.in_sea(nn[0], nn[1]):
                        has_bad_n = True
                        break
                has_bad_nn = False
                if has_bad_n is False:
                    newres = Planner.get_nears(n[0], n[1])
                    for nn in newres:
                        nnewres = Planner.get_nears(nn[0], nn[1])
                        for nnn in nnewres:
                            if not self.geo_image.in_sea(nnn[0], nnn[1]):
                                has_bad_nn = True
                                break
                if has_bad_n:
                    bad_n.append(3)
                elif has_bad_nn:
                    bad_n.append(1.5)
                else:
                    bad_n.append(1)
            for n, z in zip(res, bad_n):
                nx = round(n[0], Planner.rounder)
                ny = round(n[1], Planner.rounder)
                ll = LineString([[x, y], [nx, ny]]).buffer(0.0000001)
                # if self.geo_image.in_sea(nx, ny):
                if any(self.geo_image.geo.intersects(ll)) is False:
                    nnn = self.normalize(Vector2D(nx, ny))
                    number_n = self.pos2key[nnn]
                    self.graph.append([number, number_n, Vector2D(x, y).dist(Vector2D(nx, ny)) * z])

    def show_graph(self):
        tmp = self.geo_image.geo.plot()
        plt.xlim(x_lim)
        plt.ylim(y_lim)

        points = []
        lines = []
        weights = []
        crs = {'init': 'epsg:4326'}
        max_w = max([g[2] for g in self.graph])
        for g in self.graph:
            start_number = g[0]
            end_number = g[1]
            weight = g[2] / max_w
            weights.append(weight)
            start_pos = self.key2pos[start_number]
            end_pos = self.key2pos[end_number]
            p1 = Point(start_pos).buffer(0.0004)
            p2 = Point(end_pos).buffer(0.0004)
            points.append(p1)
            points.append(p2)
            ll = LineString([start_pos, end_pos]).buffer(0.0004)
            lines.append(ll)

        polygon = gpd.GeoDataFrame(index=list(range(len(lines))), crs=crs, geometry=lines)
        polygon.plot(color=[[w, 0.1, 0.1] for w in weights], ax=tmp, alpha=0.75)
        polygon = gpd.GeoDataFrame(index=list(range(len(points))), crs=crs, geometry=points)
        polygon.plot(color='b', ax=tmp)
        plt.show()

    def floyd_warshall(self, n, edge):
        n = max(self.key2pos.keys())
        rn = range(n + 1)
        dist = [[inf] * (n + 1) for i in rn]
        nxt = [[-1] * (n + 1) for i in rn]
        for i in rn:
            dist[i][i] = 0
        for u, v, w in edge:
            dist[u][v] = w
            nxt[u][v] = v
        last_k = 0
        for k, i, j in product(rn, repeat=3):
            if k > last_k:
                last_k = k
                print('floyd:', last_k / n * 100)
            sum_ik_kj = dist[i][k] + dist[k][j]
            if dist[i][j] > sum_ik_kj:
                dist[i][j] = sum_ik_kj
                if dist[i][j] != inf:
                    nxt[i][j] = nxt[i][k]
        # for n in nxt:
        #     print(n)
        # print("pair     dist    path")
        # for i, j in product(rn, repeat=2):
        #     if i != j:
        #         path = [i]
        #         find = False
        #         while path[-1] != j:
        #             n = nxt[path[-1]][j]
        #             if n == -1:
        #                 break
        #             path.append(n)
        #
        #         if path[-1] == j:
        #             find = True
        #         if find:
        #             print("%d → %d  %4d       %s"
        #                   % (i + 1, j + 1, dist[i][j],
        #                      ' → '.join(str(p + 1) for p in path)))
        self.nxt = nxt

    def get_path(self, start, target):
        start = self.normalize(start)
        target = self.normalize(target)
        start_num = self.pos2key[start]
        target_num = self.pos2key[target]
        path = [start_num]
        find = False
        while path[-1] != target_num:
            n = self.nxt[path[-1]][target_num]
            if n == -1:
                break
            path.append(n)

        if path[-1] == target_num:
            find = True

        path_pos = []
        if find:
            for pn in path:
                path_pos.append(self.key2pos[pn])
        return path_pos

    def normalize(self, pos):
        new_pos = None
        dist = 1000
        for n in self.pos2key.keys():
            d = Vector2D(n[0], n[1]).dist(pos)
            if d < dist:
                dist = d
                new_pos = n
        return new_pos

    @staticmethod
    def get_nears(x, y):
        res = [(x + Planner.x_dist, y),
                (x - Planner.x_dist, y),
                (x, y + Planner.y_dist),
                (x, y - Planner.y_dist),
                (x + Planner.x_dist, y + Planner.y_dist),
                (x + Planner.x_dist, y - Planner.y_dist),
                (x - Planner.x_dist, y + Planner.y_dist),
                (x - Planner.x_dist, y - Planner.y_dist)]
        for p in planner_helper_points:
            if p[0] != x and p[1] != y:
                if Vector2D(x, y).dist(Vector2D(p[0], p[1])) < 2 * Planner.x_dist:
                    res.append((p[0], p[1]))
        return res

    @staticmethod
    def douglas_peucker(PointList, epsilon):
        if len(PointList) == 1:
            return PointList
        d_max = 0
        index = 0
        end = len(PointList) - 1
        for i in range(1, end):
            d = Line2D(Vector2D(PointList[0][0], PointList[0][1]), Vector2D(PointList[end][0], PointList[end][1])).dist(
                Vector2D(PointList[i][0], PointList[i][1]))
            if d > d_max:
                index = i
                d_max = d

        if d_max > epsilon:
            rec_results1 = Planner.douglas_peucker(PointList[0:index], epsilon)
            rec_results2 = Planner.douglas_peucker(PointList[index:end + 1], epsilon)
            result_list = rec_results1[:-1] + PointList[index:index + 1] + rec_results2[1:]
        else:
            result_list = PointList[0:1] + PointList[-1:]
        return result_list

    @staticmethod
    def ploter(p, start, path, posfix_name=''):
        tmp = p.geo_image.geo.plot(color=Color.ground.value[0])
        tmp.set_facecolor(Color.sea.value[0])

        static_points = []
        for o in p.nodes:
            point = Point(o.x(), o.y()).buffer(0.002)
            static_points.append(point)
        static_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(static_points))], crs={'init': 'epsg:4326'},
                                             geometry=static_points)
        static_points_gdf.plot(ax=tmp, color='r')

        path_points = []
        for o in path:
            point = Point(o[0], o[1]).buffer(0.0005)
            path_points.append(point)
        path_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(path_points))], crs={'init': 'epsg:4326'},
                                           geometry=path_points)
        path_points_gdf.plot(ax=tmp, color='b')
        plt.plot([x[0] for x in path], [x[1] for x in path], 'r', markersize=0.3)

        sea_point = []
        for o in p.pos2key.keys():
            point = Point(o[0], o[1]).buffer(0.0001)
            sea_point.append(point)
        path_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(sea_point))], crs={'init': 'epsg:4326'},
                                           geometry=sea_point)
        path_points_gdf.plot(ax=tmp, color='g')
        plt.show()


if __name__ == "__main__":
    p = Planner(use_here=True, just_show_graph=False)
    for i in range(10):
        s = random.randint(0, len(p.key2pos.keys()))
        t = random.randint(0, len(p.key2pos.keys()))
        start = Vector2D(p.key2pos[s][0], p.key2pos[s][1])
        target = Vector2D(p.key2pos[t][0], p.key2pos[t][1])
        path = p.get_path(start, target)
        Planner.ploter(p, start, path)
        dpath = Planner.douglas_peucker(path, 0.1)
        Planner.ploter(p, start, dpath, 'Douglas0.1')
        dpath = Planner.douglas_peucker(path, 0.01)
        Planner.ploter(p, start, dpath, 'Douglas0.01')
        dpath = Planner.douglas_peucker(path, 0.001)
        Planner.ploter(p, start, dpath, 'Douglas0.001')
