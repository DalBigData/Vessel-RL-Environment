import matplotlib.pyplot as plt
import numpy as np
import fnmatch
import os

def read_file(path):
    file = open(path, 'r')
    lines = file.readlines()

    arrive_list = []
    obs_list = []
    land_list = []
    hit_list = []
    other_list = []
    vanish_list = []

    avg_number = 2000
    if path.find('plan') >= 0:
        avg_number = 100

    arrive = 0
    obs = 0
    land = 0
    other = 0
    vanish = 0

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('Other'):
            other += 1
        elif line.startswith('Arr'):
            arrive += 1
        elif line.startswith('Obs'):
            obs += 1
        elif line.startswith('Vanish'):
            vanish += 1
        elif line.startswith('Col') or line.startswith('Land'):
            land += 1

        if (i - 1) % avg_number == 0 and i > 0:
            arrive_list.append(arrive * 100 / avg_number)
            obs_list.append(obs * 100 / avg_number)
            land_list.append(land * 100 / avg_number)
            other_list.append(other * 100 / avg_number)
            vanish_list.append(vanish * 100 / avg_number)
            hit_list.append((obs + land) * 100 / avg_number)
            arrive = 0
            obs = 0
            land = 0
            other = 0
            vanish = 0
    return arrive_list

base_path = '/home/nader/workspace/dal/run_res/run_res7/'
data_dic = {
    '0.01': [['res1', 'res2', 'res3', 'res4'], []],
    '0.02': [['res5', 'res6', 'res7', 'res8'], []],
    '0.04': [['res9', 'res10', 'res11', 'res12'], []],
    '0.08': [['res13', 'res14', 'res15', 'res16'], []],
}
color = {
    '0.01': 'blue',
    '0.02': 'red',
    '0.04': 'orange',
    '0.08': 'green'
}
use_regex = True

for d in data_dic.keys():
    for p in data_dic[d][0]:
        path = base_path + p + '/'
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*happen'):
                path = path + file
                res = read_file(path)
                data_dic[d][1].append(res)
    res_average = []
    res_min = []
    res_max = []
    for i in range(len(data_dic[d][1][0])):
        s = 0
        mini = 100
        maxi = 0
        for h in range(len(data_dic[d][1])):
            s += data_dic[d][1][h][i]
            if data_dic[d][1][h][i] > maxi:
                maxi = data_dic[d][1][h][i]
            if data_dic[d][1][h][i] < mini:
                mini = data_dic[d][1][h][i]
        s /= len(data_dic[d][1])
        res_average.append(s)
        res_min.append(mini)
        res_max.append(maxi)


    data_dic[d].append(res_min)
    data_dic[d].append(res_average)
    data_dic[d].append(res_max)

for d in data_dic.keys():
    plt.plot([i for i in range(len(data_dic[d][2]))], data_dic[d][3], lw=2, label='mean population 1', color=color[d])
    plt.fill_between([i for i in range(len(data_dic[d][2]))], data_dic[d][2], data_dic[d][4], facecolor=color[d], alpha=0.5)
    print(max(data_dic[d][4]))
plt.ylim([0, 100])
plt.title('Method 1')
plt.xlabel('train episode number * 1000')
plt.ylabel('arrive to target percent')
plt.show()

