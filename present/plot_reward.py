#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt
from os import path
from typing import List
import numpy as np


def read_reward_file(in_path, step=1000) -> List[float]:
    lines = open(in_path, 'r').readlines()

    rewards = []
    for line in lines:
        rewards.append(float(line))

    size = len(rewards)

    avg_rewards = []
    steps = []
    for s in range(int(size / step)):
        sum = 0
        for r in range(s * step, (s + 1) * step):
            sum += rewards[r]
        avg_rewards.append(sum / step)
        steps.append(s * step)
    print(avg_rewards)
    return avg_rewards


if __name__ == '__main__':
    in_path_first = '/home/nader/workspace/dal/res1/2020-00-10-21-00-56_train_episode_reward_ag0'
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    average_number = 100
    first_list = read_reward_file(in_path_first, average_number)

    in_path_second = in_path_first.replace('train', 'test')
    is_second = False
    if path.isfile(in_path_second):
        is_second = True
        second_list = read_reward_file(in_path_second, average_number)
        print(len(first_list), np.argmax(np.array(second_list)), max(second_list))
        print(len(second_list), np.argmax(np.array(second_list)), max(second_list))

        plt.plot([i for i in range(len(first_list))], first_list, 'b')
        z = len(first_list) / (len(second_list) - 1)
        print([i * z for i in range(len(second_list))])
        plt.plot([i * z for i in range(len(second_list))], second_list, 'r--')
    else:
        plt.plot([i for i in range(len(first_list))], first_list, 'b')

    plt.show()
