""" generate the training and test datasets
"""
#! usr/bin/env python3

import numpy as np
import dataRead
import dataVectorization
import dataPartition

# files to be read
FILE_PATH = "../../data/miRBase_set.csv"
FILE_PATH_PUTATIVE = "../../data/putative_mirtrons_set.csv"

# read datasets and merge as a numpy array
all_data_array = dataRead.read_data(FILE_PATH,FILE_PATH_PUTATIVE)

# vectorization of the dataset
vectorized_dataset = dataVectorization.vectorize_data(all_data_array)

# partition the dataset
X_train, y_train, X_test, y_test = dataPartition.data_partition(vectorized_dataset)

# transform to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


