#! /usr/bin/env python3

"""predict whether it is mirtron.  "yes" means "is mirtron".
   "no" means "is not a mirtron"
"""

import tensorflow as tf
import sys
sys.path.append("./src/data_preprocess")
sys.path.append("./src/model_construct")
sys.path.append("../model_evaluate")
import dataRead
import dataVectorization
import model_construction
import getopt
import numpy as np
 
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


# placeholder
#input_X = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH,EMBEDDING_SIZE,1])
#input_y = tf.placeholder(tf.float32,[None, NUM_CLASSES])
#keep_prob = tf.placeholder(tf.float32)

x_cast = {"A":[[1],[0],[0],[0]],"U":[[0],[1],[0],[0]],\
          "T":[[0],[1],[0],[0]],"G":[[0],[0],[1],[0]],\
          "C":[[0],[0],[0],[1]],"N":[[0],[0],[0],[0]]}

def usage():
    print("USAGE: python isMirtron -s pre-miRNA sequnece\n")
    print("Example: python isMirtron -s CTGGGGAGATGGGGGGAGCTCTGCTGAGGGTGCACAAGGCCCTGGCTCTACACACATCCCTGTCTTACAG")

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
def seq_vectorize(seq):
    vectorized_seq = []
    for char in processed_seq:
        vectorized_seq.append(x_cast[char])
    vectorized_seq = np.array(vectorized_seq)
    vectorized_seq = vectorized_seq.reshape([1,SEQUENCE_LENGTH,EMBEDDING_SIZE,1])
    return vectorized_seq

try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:",["help","sequence="])
except getopt.GetoptError:
    print ("Wrong usage!\n")
    usage()
    sys.exit(1)
if len(opts) < 1:
    usage()
    sys.exit(1)
# parse the options
for op, value in opts:
    if op in ("-s","--sequence"):
        input_seq = value
    elif op in ("-h","--help"):
        usage()
        sys.exit()
print("Prediction result:")
print("Your input sequence:",input_seq)
# padding and vectorize the input sequence
processed_seq = seq_process(input_seq)
vectorized_seq = seq_vectorize(processed_seq)

#print("vectorized_seq shape",vectorized_seq.shape)

# import the  trained model parameter

# calculate the prediction result using the trained model
#print("sess start:")
with tf.Session() as sess:
    # restore the trained model
    saver = tf.train.import_meta_graph('logs/filter6/filter6.ckpt.meta')
    # print("graph restore succeed")
    saver.restore(sess,tf.train.latest_checkpoint("logs/filter6/"))
    # print("parameters restore succeed")

    predict_result = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()


    # get the placeholder in the restored graph by the method of 
    # get_operation_by_name()
    input_X = graph.get_operation_by_name('input_X').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    # feed the new data
    # print("feed the data start")
    prediction = sess.run(predict_result,feed_dict={input_X:vectorized_seq,\
                                       keep_prob:1})

    # print("feed the data end")
    # print ("prediction:",prediction)
    # print the predicted class
    m_index = sess.run(tf.argmax(prediction,1))
    if m_index == 0:
        print("Yes,it is a mirtron.")
    elif m_index == 1:
        print("No,it is not a mirtron.")


