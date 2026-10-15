""" The common components which can be used by all the models
"""

import tensorflow as tf

#hyperparameters
LR = 0.001       #learning rate
TRAINING_ITER = 300   #iteration times
BATCH_SIZE = 18        #batch size of input

SEQUENCE_LENGTH = 164   #sequence length of input
EMBEDDING_SIZE = 4      #char embedding size(sequence width of input)

#CONV_SIZE = 3    #first filter size
#CONV_DEEP = 128   #number of first filter(convolution deepth)

STRIDES = [1,1,1,1]  #the strid in each of four dimensions during convolution
KSIZE = [1,164,1,1]    #pooling window size

FC_SIZE = 128     #nodes of full-connection layer
NUM_CLASSES = 2   # classification number

DROPOUT_KEEP_PROB = 0.5   #keep probability of dropout

# define placeholder
input_X = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH,EMBEDDING_SIZE,1])
input_y = tf.placeholder(tf.float32,[None, NUM_CLASSES])
#keep_prob = tf.placeholder(tf.float32) 

# function of initialize weights
def get_weights_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)
    weights = tf.Variable(initial,name = "weights")
    return weights

# function of convolution and pooling
def conv_and_pooling(input_tensor,filter_height,filter_width,\
                     depth,conv_deep,layer_name):
    
    with tf.name_scope(layer_name):
        conv_weights = get_weights_variable\
                    ([filter_height,filter_width,depth,conv_deep])
        conv_bias = tf.Variable(tf.constant(0.1,shape=[conv_deep]),name = "bias")   
        conv = tf.nn.conv2d(input_tensor,conv_weights,strides = STRIDES,\
                            padding='SAME')
        conv_relu = tf.nn.relu(tf.nn.bias_add(conv,conv_bias))
        conv_relu_pool = tf.nn.max_pool(conv_relu,ksize=KSIZE,\
                                        strides=STRIDES,padding='VALID')
        # tensorboard visualization
        tf.summary.histogram("../../data/conv_weights",conv_weights)
        tf.summary.histogram("../../data/conv_bias",conv_bias)
        return conv_relu_pool

def fc_output_inference(input_tensor,fc_size,output_size,keep_prob):
    shape_list = input_tensor.get_shape().as_list()
    nodes = shape_list[1]*shape_list[2]*shape_list[3]
    reshaped = tf.reshape(input_tensor,[-1,nodes])

    # the first fully connected layer
    fc1_weights = get_weights_variable([nodes,fc_size])
    fc1_bias = tf.Variable(tf.constant(0.1,shape=[fc_size]))
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_bias)
    
    # avoid overfitting, droupout regularization
    fc1 = tf.nn.dropout(fc1,keep_prob)
    #fc1 = tf.nn.dropout(fc1,0.5)
 
    # the second fully connected layer(output layer)
    fc2_weights = get_weights_variable([fc_size,output_size])
    fc2_bias = tf.Variable(tf.constant(0.1,shape=[output_size]))
    output = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_bias)

    return output
 
# use only one kinds of filter in the CNN structure
def cnn_mono_inference(input_tensor,filter_height,filter_width,\
                  in_channels,out_channels,layer_name,keep_prob):
    # layer of convolution and max-pooling
    conv_pool = conv_and_pooling(input_tensor,filter_height,\
                                  filter_width,1,out_channels,layer_name)
 
    output = fc_output_inference(conv_pool,FC_SIZE,NUM_CLASSES,keep_prob)
    
    return output

# use different sizes of filters in the CNN structure
def cnn_concat_inference(input_tensor,filter_height_list,filter_width,\
                         in_channels,out_channels,layer_name_list,keep_prob):
    conv_pool_list = []
    filter_num = len(filter_height_list)
    for i in range(filter_num):
        conv_pool = conv_and_pooling(input_tensor,filter_height_list[i],filter_width,\
                                     in_channels,out_channels,layer_name_list[i]) 
        conv_pool_list.append(conv_pool)

    # concatenant all the conv_pools tensor
    conv_pool_concat = tf.concat([conv_pool for conv_pool in conv_pool_list],1)

    output = fc_output_inference(conv_pool_concat,FC_SIZE,NUM_CLASSES,keep_prob)

    return output


if __name__ == "__main__":
    
    cnn_mono_inference(input_X,3,EMBEDDING_SIZE,1,128,"Conv3-128")
    print ("model Conv3-128 was constructed!")
    cnn_concat_inference(input_X,[3,4,5,6],EMBEDDING_SIZE,1,32,\
                         ["Conv3-32","Conv4-32","Conv5-32","Conv6-32"])
    print ("model Conv-concat was constructed!")
