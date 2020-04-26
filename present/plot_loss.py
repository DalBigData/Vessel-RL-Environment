import matplotlib.pyplot as plt

file = '/home/nader/workspace/dal/run_res/run_res2/res6/agent_loss'
file = open(file, 'r')
file = file.readlines()
loss = []
for l in file:
    loss.append(float(l))
# loss = loss[:30000]
plt.plot([i for i in range(len(loss))], loss)
plt.show()
print(min(loss))