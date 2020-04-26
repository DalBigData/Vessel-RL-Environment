from math import inf


def floyd_warshall(n, edge):
    rn = range(n)
    dist = [[inf] * n for i in rn]
    nxt = [[-1] * n for i in rn]
    for i in rn:
        dist[i][i] = 0
    for u, v, w in edge:
        dist[u - 1][v - 1] = w
        nxt[u - 1][v - 1] = v - 1
    last_k = 0
    # from itertools import product
    # for k, i, j in product(rn, repeat=3):

    for k in range(n):
        for i in range(n):
            for j in range(n):
                sum_ik_kj = dist[i][k] + dist[k][j]
                if dist[i][j] > sum_ik_kj:
                    dist[i][j] = sum_ik_kj
                    nxt[i][j] = nxt[i][k]
    # print(nxt)
edge = [(1,2,1), (1,4,1), (2,1,1),(2,5,1), (2,3,1),(3,2,1),(3,6,1), (4,1,1),(4,5,1),(4,7,1), (5,2,1),(5,4,1),(5,8,1),(5,6,1),(6,3,1),(6,5,1),(6,9,1),(7,4,1),(7,8,1),(8,7,1),(8,5,1),(8,9,1),(9,8,1),(9,6,1)]
import time
a = time.time()

for i in range(10000):
    floyd_warshall(9,edge)

b = time.time()
print(b - a)