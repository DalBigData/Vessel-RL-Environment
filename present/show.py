import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point
import geopandas as gpd
import sys
from present.color import Color
from enum import Enum
import os.path
import time


class Mode(Enum):
    pic = 0,
    gif = 1,
    show = 2


in_path = '/home/nader/workspace/dal/res1/'
out_path = '/home/nader/workspace/dal/res_out/'
file = '2020-41-14-20-41-38_9000_1_'
mode = Mode.pic
read_plan = True
show_local_view = True

if len(sys.argv) > 1:
    in_path = sys.argv[1]
    file = sys.argv[2]

if len(sys.argv) > 3:
    if sys.argv[3] == 'gif':
        mode = Mode.gif
    elif sys.argv[3] == 'pic':
        mode = Mode.pic

path = in_path + file

x_lim = None
y_lim = None
map_path = ''
crs = {}
static_object = []
agent_r = 0
target_r = 0
last_target = []

dynamic_object = []
ais_object = []
target = []
agent_pos = []
local_view = []


def read(file_path):
    global x_lim, y_lim, map_path, crs, static_object, dynamic_object, ais_object, agent_r, target_r, target,\
        agent_pos, last_target
    lines = open(file_path, 'r').readlines()
    for line in lines:
        if line.find('size:') >= 0:
            tmp = line[5:]
            x_lim = eval(tmp)[0]
            y_lim = eval(tmp)[1]
        elif line.find('path:') >= 0:
            map_path = line[5:-1]
        elif line.find('static_object:') >= 0:
            static_object = eval(line[14:])
        elif line.find('dynamic_object:') >= 0:
            dynamic_object.append(eval(line[15:]))
        elif line.find('ais_object:') >= 0:
            ais_object.append(eval(line[11:]))
        elif 0 <= line.find('target:') <= 3:
            target_r = eval(line[7:])[0]
            target.append(eval(line[7:])[1])
        elif line.find('last_target:') >= 0:
            last_target = eval(line[12:])[1]
        elif line.find('agent:') >= 0:
            agent_pos.append(eval(line[6:])[1])
            agent_r = eval(line[6:])[0]
        elif line.find('crs:') >= 0:
            crs = eval(line[4:])


def read_local_view(episode_file):
    step_number = 0
    views = []
    while True:
        local_view_name = episode_file + 'localview' + str(step_number)
        if not os.path.isfile(local_view_name):
            break
        first_line = open(local_view_name, 'r').readline()
        view = eval(first_line)['view']
        views.append(view)
        step_number += 1
    return views


if not read_plan:
    read(path)
    if show_local_view:
        local_view.append(read_local_view(path))
else:
    episode_number = 1
    while True:
        name = path + str(episode_number) + '_0'
        if not os.path.isfile(name):
            break
        read(name)
        if show_local_view:
            local_view.append(read_local_view(name))
        episode_number += 1

if show_local_view:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax1 = axs[1]
    ax = axs[0]
else:
    fig, axs = plt.subplots()
    ax = axs

print('../' + map_path)
gpd_map = gpd.read_file('../' + map_path).to_crs(crs)
ln, = ax.plot([], [], 'ro')
ax.set_xlim(x_lim[0], x_lim[1])
ax.set_ylim(y_lim[0], y_lim[1])

step = 0
episode = 0

way_size = min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) / 500

pre_agent_positions = []
pre_agent_targets = []

tt = []


def update(frame):
    global step, ax, episode, pre_agent_positions, pre_agent_targets, ax1, local_view, tt
    t1 = time.time()
    print(len(agent_pos), step, frame)
    # sea
    base_x = [x_lim[0], x_lim[1], x_lim[1], x_lim[0]]
    base_y = [y_lim[1], y_lim[1], y_lim[0], y_lim[0]]
    base_pol = Polygon(zip(base_x, base_y))
    base_pol = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[base_pol])
    ax = base_pol.plot(ax=ax, color=Color.sea.value[0])

    # ground
    ax = gpd_map.plot(ax=ax, color=Color.ground.value[0])

    # target
    p = Point(target[episode][0], target[episode][1]).buffer(target_r)
    target_point = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[p])

    if step == 0:
        pre_agent_targets.append(p)
    if episode > 1:
        previous_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(pre_agent_targets) - 1)],
                                               crs={'init': 'epsg:4326'}, geometry=pre_agent_targets[:-1])
        ax = previous_points_gdf.plot(ax=ax, color=Color.sea.value[0], edgecolor=[211 / 255, 211 / 255, 211 / 255],
                                      markersize=1)

    ax = target_point.plot(ax=ax, color=Color.target.value[0])
    p = Point(last_target[0], last_target[1]).buffer(target_r)
    target_point = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[p])
    ax = target_point.plot(ax=ax, color=Color.last_target.value[0])

    # static objects
    static_points = []
    for o in static_object:
        r = o[0]
        pos = o[1]
        p = Point(pos[0], pos[1]).buffer(r * 3.0)
        static_points.append(p)

    static_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(static_points))], crs={'init': 'epsg:4326'},
                                         geometry=static_points)
    ax = static_points_gdf.plot(ax=ax, color="orange")

    # dynamic objects
    for o in dynamic_object:
        if step > 0:
            previous_points = [Point(pos[0], pos[1]).buffer(way_size) for pos in o[1][:step]]
            previous_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(previous_points))],
                                                   crs={'init': 'epsg:4326'}, geometry=previous_points)
            ax = previous_points_gdf.plot(ax=ax, color=Color.dynamic_object.value[0])
        r = o[0]
        pos = o[1][step]
        last_point = Point(pos[0], pos[1]).buffer(r * 3.0)
        last_point = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[last_point])
        ax = last_point.plot(ax=ax, color=Color.dynamic_object.value[0])

    # ais objects
    for o in ais_object:
        if step > 0:
            previous_points = [Point(pos[0], pos[1]).buffer(way_size) for pos in o[1][:step]]
            previous_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(previous_points))],
                                                   crs={'init': 'epsg:4326'}, geometry=previous_points)
            ax = previous_points_gdf.plot(ax=ax, color=Color.ais_object.value[0])
        r = o[0]
        pos = o[1][step]
        last_point = Point(pos[0], pos[1]).buffer(r * 3.0)
        last_point = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[last_point])
        ax = last_point.plot(ax=ax, color=Color.ais_object.value[0])

    # agent
    if step > 0 or episode > 0:
        previous_points = [Point(pos[0], pos[1]).buffer(way_size) for pos in pre_agent_positions]
        previous_points_gdf = gpd.GeoDataFrame(index=[i for i in range(len(pre_agent_positions))],
                                               crs={'init': 'epsg:4326'}, geometry=previous_points)
        ax = previous_points_gdf.plot(ax=ax, color=[1, 99 / 255, 71 / 255])
    print(agent_pos)
    print(episode, step)
    pos = agent_pos[episode][step]
    pre_agent_positions.append(pos)
    last_point = Point(pos[0], pos[1]).buffer(agent_r*3)
    last_point = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[last_point])
    ax = last_point.plot(ax=ax, color=[1, 99 / 255, 71 / 255])

    # local view
    if show_local_view:
        ax1.cla()
        ax1.set_xlim([0, 50])
        ax1.set_ylim([0, 50])
        print('hh', episode, step)
        print('hh', len(local_view[episode]))
        local_view_all_x = [[i for _ in range(51)] for i in range(51)]
        local_view_all_y = [[j for j in range(51)] for _ in range(51)]

        local_view_ground_x = []
        local_view_ground_y = []

        local_view_obs_x = []
        local_view_obs_y = []

        local_view_target_x = []
        local_view_target_y = []

        for x in range(len(local_view[episode][step])):
            for y in range(len(local_view[episode][step][0])):
                if 0.49 < local_view[episode][step][x][y] < 0.51:
                    local_view_ground_x.append(x)
                    local_view_ground_y.append(y)
                if local_view[episode][step][x][y] > 0.8:
                    local_view_target_x.append(x)
                    local_view_target_y.append(y)
                if 0.29 < local_view[episode][step][x][y] < 0.31:
                    local_view_obs_x.append(x)
                    local_view_obs_y.append(y)

        ax1.plot(local_view_all_x, local_view_all_y, 'o', color='white', markersize=12)
        ax1.plot(local_view_target_x, local_view_target_y, 'o', color='green', markersize=7)
        ax1.plot(local_view_ground_x, local_view_ground_y, 'o', color='saddlebrown', markersize=7)
        ax1.plot(local_view_obs_x, local_view_obs_y, 'o', color='orange', markersize=7)
        ax1.plot([25], [25], 'o', color=Color.agent.value, markersize=8)
    step += 1
    if step >= len(agent_pos[episode]):
        step = 0
        episode += 1
    tt.append(time.time() - t1)
    return ln,


number_of_step = 0
for ep in agent_pos:
    number_of_step += len(ep)


def output_path():
    global out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)


if mode == Mode.show:
    ani = FuncAnimation(fig, update, blit=False, interval=100, frames=number_of_step - 1, repeat=False)
    plt.show()
if mode == Mode.gif:
    output_path()
    ani = FuncAnimation(fig, update, blit=False, interval=1, frames=number_of_step - 1, repeat=False)
    ani.save(out_path + file + '.gif', writer='imagemagick', fps=2, dpi=200)
if mode == Mode.pic:
    output_path()
    step = len(agent_pos[0]) - 1
    update(0)
    plt.show()
    fig.savefig(out_path + file)

print(tt)
