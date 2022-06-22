import numpy as np


fname = "mnist_train.csv"
raw_data = np.loadtxt(fname, delimiter=",")

reduction = 10
data = raw_data[::reduction]

savestring = fname[:-4] + "_reduced" + str(reduction) + ".csv"
np.savetxt(savestring, data, delimiter=",")

