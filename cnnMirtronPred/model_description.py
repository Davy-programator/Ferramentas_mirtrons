#! /usr/bin/env python3

# Use CSV with id and seq headers
# See README.md
# Use with conda environment
# conda run -p .condaenv python isMirtronCsv.py --csv datac/samples1.csv

"""predict whether it is mirtron.  "1" means "is mirtron".
   "0" means "is not a mirtron"
"""

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
sys.path.append("./src/data_preprocess")
sys.path.append("./src/model_construct")
sys.path.append("../model_evaluate")
import getopt
import numpy as np
import csv

BATCH_SIZE_RESULT = 1000

# hyperparameters
#LR = 0.001       #learning rate
#TRAINING_ITER = 10000   #iteration times
#BATCH_SIZE = 18        #batch size of input
SEQUENCE_LENGTH = 164   #sequence length of input
EMBEDDING_SIZE = 4      #char embedding size(sequence width of input)

STRIDES = [1,1,1,1]  #the strid in each of four dimensions during convolution
KSIZE = [1,164,1,1]    #pooling window size
 
FC_SIZE = 128     #nodes of full-connection layer
NUM_CLASSES = 2   # classification number

x_cast = {"A":[[1],[0],[0],[0]],"U":[[0],[1],[0],[0]],\
          "T":[[0],[1],[0],[0]],"G":[[0],[0],[1],[0]],\
          "C":[[0],[0],[0],[1]],"N":[[0],[0],[0],[0]]}

def create_tf_session():
  sess = tf.Session(config=sess_config)

  # restore the trained model
  saver = tf.train.import_meta_graph('{modelpath}/{modelname}.meta'.format(modelpath=opts['modelpath'], modelname=opts['modelname']))
  saver.restore(sess,tf.train.latest_checkpoint(opts['modelpath']))

  predict_result = tf.get_collection('pred_network')[0]
  graph = tf.get_default_graph()

  input_X = graph.get_operation_by_name('input_X').outputs[0]
  keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

  return sess, predict_result, input_X, keep_prob

def process_argv():
  r = {}
  r['modelpath'] = "logs/filter6"
  r['modelname'] = 'filter6.ckpt'
  return r

opts = process_argv()

# Session preparation
sess = None
sess_config = tf.ConfigProto() 
sess_config.gpu_options.allow_growth = True 

tf.reset_default_graph()
sess, predict_result, input_X, keep_prob = create_tf_session()

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    # print(len(shape))
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)

# print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
