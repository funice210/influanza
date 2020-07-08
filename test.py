
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

plt.rc('figure', figsize=(8.0, 8.0))

data, label = make_blobs(n_samples=200, random_state=0)
label = label.reshape(200, 1)
plt.scatter(data[:,0], data[:,1], s=20, c=label, cmap=plt.cm.Accent) 