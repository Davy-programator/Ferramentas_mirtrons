""" cnn model with one-layer of convolution (Conv3-128)
"""

import model_shared
import tensorflow as tf

def model_conv3_output(input_X,EMBEDDING_SIZE,keep_prob):
    return model_shared.cnn_mono_inference(input_X,3,EMBEDDING_SIZE,\
                                           1,128,"Conv3-128",keep_prob)

def model_conv4_output(input_X,EMBEDDING_SIZE,keep_prob):
    return model_shared.cnn_mono_inference(input_X,4,EMBEDDING_SIZE,\
                                           1,128,"Conv4-128",keep_prob)

def model_conv5_output(input_X,EMBEDDING_SIZE,keep_prob):
    return model_shared.cnn_mono_inference(input_X,5,EMBEDDING_SIZE,\
                                           1,128,"Conv5-128",keep_prob)


def model_conv6_output(input_X,EMBEDDING_SIZE,keep_prob):
    return model_shared.cnn_mono_inference(input_X,6,EMBEDDING_SIZE,\
                                           1,128,"Conv6-128",keep_prob)

def model_concat_output(input_X,EMBEDDING_SIZE,keep_prob):
    concat_output = model_shared.cnn_concat_inference\
                         (input_X,[3,4,5,6],EMBEDDING_SIZE,1,32,\
                          ["Conv3-32","Conv4-32","Conv5-32","Conv6-32"],keep_prob)
    return concat_output


if __name__ == "__main__":
    # hyperparameters
    LR = 0.001       #learning rate
    TRAINING_ITER = 3000   #iteration times
    BATCH_SIZE = 180        #batch size of input

    SEQUENCE_LENGTH = 164   #sequence length of input
    EMBEDDING_SIZE = 4      #char embedding size(sequence width of input)
    # CONV_SIZE = 3    #first filter size
    # CONV_DEEP = 128   #number of first filter(convolution deepth)

    STRIDES = [1,1,1,1]  #the strid in each of four dimensions during convolution
    KSIZE = [1,164,1,1]    #pooling window size

    FC_SIZE = 128     #nodes of full-connection layer
    NUM_CLASSES = 2   # classification number

    DROPOUT_KEEP_PROB = 0.5   #keep probability of dropout
 
    # define placeholder
    input_X = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH,EMBEDDING_SIZE,1])
    input_y = tf.placeholder(tf.float32,[None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)
    model_conv3_output(input_X,EMBEDDING_SIZE,keep_prob)
    model_conv4_output(input_X,EMBEDDING_SIZE,keep_prob)
    model_conv5_output(input_X,EMBEDDING_SIZE,keep_prob)
    model_conv6_output(input_X,EMBEDDING_SIZE,keep_prob)
    model_concat_output(input_X,EMBEDDING_SIZE,keep_prob)
