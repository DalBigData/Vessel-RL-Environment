import sys
import time
import datetime
import matplotlib.pyplot as plt
import geopandas
# import matplotlib; matplotlib.use("TkAgg")


def ais_to_episode(path, episode_real_time):
    file = open(path, 'r')
    all_continues = []
    lines = file.readlines()[1:]
    for l in lines:
        tmp = l.split(',')
        str_time = tmp[1]
        str_time = str_time.replace('Z', '000')
        epoch = time.mktime(datetime.datetime.strptime(str_time, "%Y-%m-%dT%H:%M:%S.%f").timetuple())
        tmp = [int(epoch), float(tmp[2]), float(tmp[3]), str_time]
        all_continues.append(tmp)

    min_epoch = all_continues[0][0]
    max_epoch = all_continues[-1][0]
    all_discrete = []
    epoch_time = min_epoch
    epoch_number = 0
    episode_time = episode_real_time
    while True:
        if epoch_number >= len(all_continues):
            break
        if epoch_time <= all_continues[epoch_number][0]:

            if epoch_number == 0:
                all_discrete.append([all_continues[0][1], all_continues[0][2]])
            else:
                e2 = all_continues[epoch_number][0]
                x2 = all_continues[epoch_number][1]
                y2 = all_continues[epoch_number][2]
                e1 = all_continues[epoch_number - 1][0]
                x1 = all_continues[epoch_number - 1][1]
                y1 = all_continues[epoch_number - 1][2]
                ze = (epoch_time - e1) / (e2 - e1)
                x = x1 + (x2 - x1) * ze
                y = y1 + (y2 - y1) * ze
                all_discrete.append([x, y])
            epoch_time += episode_time
        else:
            epoch_number += 1
    return all_discrete


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        path = args[1]
    else:
        path = 'ais_ferry_simple.csv'

    world = geopandas.read_file('../geopandas/halifax_shape/HalifaxBuffer50m2.shp')
    crs = {'init': 'epsg:4326'}
    world = world.to_crs(crs)
    tmp = world.plot()

    res = ais_to_episode(path, 100)
    f = open(path, 'r').readlines()
    xorg = []
    yorg = []
    for l in f[1:]:
        line = l.split(',')
        xorg.append(float(line[2]))
        yorg.append(float(line[3]))
        print(xorg[-1], yorg[-1])
    x = []
    y = []
    for r in res:
        x.append(r[0])
        y.append(r[1])
        print(x[-1], y[-1])
    print(len(x))
    plt.plot(x, y, 'b-')
    plt.plot(x, y, 'r.')
    # plt.plot(xorg, yorg, 'r--')
    plt.xlim([-63.69, -63.49])
    plt.ylim([44.58, 44.73])
plt.show()
