import matplotlib.pyplot as plt
import numpy as np
import fnmatch
import os

use_regex = True

if use_regex:
    path = '/home/nader/workspace/dal/res1/'
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, '*happen_plan'):
            path = path + file
            print(file)
else:
    path = '/home/nader/workspace/dal/res6/2019-20-19-15-20-05_ha'

file = open(path, 'r')
lines = file.readlines()

arrive_list = []
obs_list = []
land_list = []
hit_list = []
other_list = []
vanish_list = []

avg_number = 200
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

fig, axs = plt.subplots(2, 3, figsize=(10, 5))

axs[0][0].plot([i * avg_number for i in range(len(land_list))], arrive_list)
axs[0][0].set_title('arrive')
axs[0][0].set_ylim([0, 100])

axs[0][1].plot([i * avg_number for i in range(len(land_list))], land_list)
axs[0][1].set_title('land')
axs[0][1].set_ylim([0, 100])

axs[0][2].plot([i * avg_number for i in range(len(land_list))], obs_list)
axs[0][2].set_title('obs')
axs[0][2].set_ylim([0, 100])

axs[1][0].plot([i * avg_number for i in range(len(land_list))], vanish_list)
axs[1][0].set_title('vanish')
axs[1][0].set_ylim([0, 100])

axs[1][1].plot([i * avg_number for i in range(len(land_list))], other_list)
axs[1][1].set_title('other')
axs[1][1].set_ylim([0, 100])

axs[1][2].plot([i * avg_number for i in range(len(land_list))], hit_list)
axs[1][2].set_title('hit')
axs[1][2].set_ylim([0, 100])

plt.show()

print(np.argmax(np.array(arrive_list)), max(arrive_list), len(arrive_list))
