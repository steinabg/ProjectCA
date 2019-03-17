import numpy as np

using_for = np.load("Data/mpi_combined_txt/Q_cbj_1000_for.npy")

using_vector = np.load("Data/mpi_combined_txt/Q_cbj_1000.npy")

list = []
list2 = []

abs_err = np.abs(using_vector-using_for)

ave = np.average((np.abs(using_for-using_vector))) # Average absolute error

# Average error in cells different from 1



for i in range(using_for.shape[0]):
    for j in range(using_for.shape[1]):
        if using_for[i,j,0] != 1:
            list.append(np.abs(using_for[i,j,0]-using_vector[i,j,0]))
        if using_for[i,j,1] != 1:
            list2.append(np.abs(using_for[i, j, 1] - using_vector[i, j, 1]))

ave2 = np.average(list)

ave3 = np.average(list2)