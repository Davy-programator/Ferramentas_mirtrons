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
# import dataRead
# import dataVectorization
# import model_construction
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
 
# DROPOUT_KEEP_PROB = 0.5   #keep probability of dropout

x_cast = {"A":[[1],[0],[0],[0]],"U":[[0],[1],[0],[0]],\
          "T":[[0],[1],[0],[0]],"G":[[0],[0],[1],[0]],\
          "C":[[0],[0],[0],[1]],"N":[[0],[0],[0],[0]]}

def usage():
  print("USAGE: python isMirtron --csv input.csv --modelpath ~/model --output output.csv")
  print("Example: python isMirtron --csv your_sequences.csv --modelpath ~/model --output output.csv")

# padding and trim the sequence to the length of SEQUENCE_LENGTH
def seq_process(seq):
  # remove all the spaces
  seq = seq.replace(' ', '')
  # remove \n
  seq = seq.replace("\n", "")
  for base in seq:
    if base not in ("A","U","G","C","T","a","u","g","c","t"):
      print ("Input sequence is wrong: incorrect base!\n")
      usage()
      exit(1)
  seq = seq.upper() 
  m_len = len(seq)
  if m_len < SEQUENCE_LENGTH:
    seq += "N" *(SEQUENCE_LENGTH - m_len)
  elif m_len >= SEQUENCE_LENGTH:
    seq = seq[:SEQUENCE_LENGTH]

  return seq

# sequence vectorization
def seq_vectorize(processed_seq):
  vectorized_seq = []
  for char in processed_seq:
    vectorized_seq.append(x_cast[char])
  vectorized_seq = np.array(vectorized_seq)
  vectorized_seq = vectorized_seq.reshape([1,SEQUENCE_LENGTH,EMBEDDING_SIZE,1])
  return vectorized_seq

def flush_results():
  csvfile_writer = open(csvoutput, 'a', newline='')
  csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
  csvwriter.writerows(results)
  csvfile_writer.close()

def create_tf_session():
  sess = tf.Session(config=sess_config)

  # restore the trained model
  saver = tf.train.import_meta_graph('{modelpath}/{modelname}.meta'.format(modelpath=opts['modelpath'], modelname=opts['modelname']))
  # print("graph restore succeed")
  saver.restore(sess,tf.train.latest_checkpoint(opts['modelpath']))
  # print("parameters restore succeed")

  predict_result = tf.get_collection('pred_network')[0]
  graph = tf.get_default_graph()

  # get the placeholder in the restored graph by the method of 
  # get_operation_by_name()
  input_X = graph.get_operation_by_name('input_X').outputs[0]
  keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

  return sess, predict_result, input_X, keep_prob

def process_argv():
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:",["help","csv=", "modelpath=", "output="])
  except getopt.GetoptError as error:
    print("Wrong usage!")
    usage()
    sys.exit(1)

  if len(opts) < 1:
    usage()
    sys.exit(1)

  # parse the options
  r = {}
  for op, value in opts:
    if op in ("-s","--csv"):
      r['csv'] = value
    if op in ("--modelpath"):
      r['modelpath'] = value
      r['modelname'] = 'model.ckpt'
    if op in ("--output"):
      r['output'] = value
    elif op in ("-h","--help"):
      usage()
      sys.exit()
  
  requireds = ["csv"]
  for required in requireds:
    if not required in r:
      print("Wrong usage!!")
      usage()
      sys.exit(1)
  
  if not "modelpath" in r:
    r['modelpath'] = "logs/filter6"
    r['modelname'] = 'filter6.ckpt'
  if not "output" in r:
    r['output'] = os.path.splitext(r['csv'])[0] + '_cnnmirtronpred.csv'

  return r

opts = process_argv()
# sys.exit(1)

input_path = opts['csv']
csvoutput = opts['output']

results = []

# Open CSV input
csvinput = open(input_path, newline='')
csvreader = csv.DictReader(csvinput, delimiter=',', quotechar='"')

# Create CSV output
fieldnames = ['id', 'seq', 'ismirtron']
# csvoutput = os.path.splitext(input_path)[0] + '_cnnmirtronpred.csv'

csvfile_writer = open(csvoutput, 'w', newline='')
csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
csvwriter.writeheader()
csvfile_writer.close()

results = []
result_counter = 0

# Session preparation
sess = None
sess_config = tf.ConfigProto() 
sess_config.gpu_options.allow_growth = True 
# sess = tf.Session(config=sess_config)

for n, row in enumerate(csvreader):

  # Session creation
  if sess is None:
    tf.reset_default_graph()
    sess, predict_result, input_X, keep_prob = create_tf_session()
  
  processed_seq = seq_process(row['seq'])
  vectorized_seq = seq_vectorize(processed_seq)

  prediction = sess.run(predict_result, feed_dict={input_X:vectorized_seq,keep_prob:1})

  # print("feed the data end")
  # print ("prediction:",prediction)
  # print the predicted class
  m_index = sess.run(tf.argmax(prediction,1))
  # if m_index == 0:
  #   print("Yes,it is a mirtron.")
  # elif m_index == 1:
  #   print("No,it is not a mirtron.")
  row['ismirtron'] = 1 if m_index == 0 else 0

  results.append(row)
  result_counter += 1

  if result_counter == BATCH_SIZE_RESULT:
    flush_results()

    results.clear()
    result_counter = 0
    sess = None

flush_results()

# Close handlers
csvinput.close()
