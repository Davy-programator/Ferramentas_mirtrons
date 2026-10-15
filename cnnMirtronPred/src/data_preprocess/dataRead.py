"""Read the positive and negative dataset from files 
"""
import numpy as np
import csv
import random

# read the datasets from csv files as a numpy array
def read_data(file1_path,file2_path):
    dataset1_reader = csv.reader(open(file1_path, encoding='utf-8'))
    dataset2_reader = csv.reader(open(file2_path, encoding='utf-8'))
    # define a list to store the data
    all_data_set = []

    # read the data into a list(name,sequence,class)
    for row in dataset1_reader:
        if row[0] == 'id': continue
        all_data_set.append([row[0],row[1],row[2]])
    for row in dataset2_reader:
        if row[0] == 'id': continue
        all_data_set.append([row[0],row[1],row[2]])    
    # shuffle the data set randomly    
    random.seed(2)
    random.shuffle(all_data_set)
    return all_data_set

if __name__ =="__main__":
    FILE_PATH = "../../data/miRBase_set.csv" 
    FILE_PATH_PUTATIVE = "../../data/putative_mirtrons_set.csv"
    all_data_list = read_data(FILE_PATH,FILE_PATH_PUTATIVE)
    print(all_data_list)
