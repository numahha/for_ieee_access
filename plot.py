import numpy as np
import matplotlib.pyplot as plt


a = []
for i in range(5):
    data = np.loadtxt("alpha0.1/data_policy_evaluation_loss_"+str(i)+".csv")
    a.append(data[:,0])
    plt.plot(data[:,0])
a = np.array(a)
print(a.shape)
plt.plot(a.mean(axis=0))
plt.show()