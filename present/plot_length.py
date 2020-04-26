import matplotlib.pyplot as plt

path = '/home/nader/workspace/dal/res1/2020-00-10-21-00-56_happen_plan_len'

x = []
y = []
file = open(path, 'r')
lines = file.readlines()
for l in lines:
    ll = l.split(' ')
    if int(ll[0]) == 0:
        continue
    x.append(int(ll[0]))
    y.append(float(int(ll[1]) / x[-1]) * 100)

plt.plot(x, y, 'ro')
plt.ylim([0,150])
plt.show()

avg = [0 for i in range(max(x))]
for i in range(len(avg)):
    s = 0
    for j in range(len(x)):
        if j == x[i]:
            s += 1
            avg[i] += y[j]
    avg[i] = avg[i] / s
plt.plot(range(len(avg)), avg, 'ro')
plt.ylim([0,150])
plt.show()
avg2 = []
print(avg)
for i in range(int(len(avg) / 10)):
    print(i)
    print(avg[i*10:(i+1)*10])
    x = sum(avg[i*10:(i+1)*10])
    print(x)
    avg2.append(x)

plt.plot(range(len(avg2)), avg2, 'ro')
plt.ylim([0,150])
plt.show()