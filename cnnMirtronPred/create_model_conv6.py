"""Train the constructed cnn model with the training dataset
"""

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
# https://stackoverflow.com/questions/60382007/how-to-lower-ram-consumption-in-tensorflow

import sys
import os

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="True"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sys.path.append("./src/data_preprocess")
sys.path.append("./src/model_construct")
sys.path.append("./src/model_evaluate")
#from model_construction import *
import dataRead
import dataVectorization
import dataPartition
import model_construction
import tensorflow as tf
import numpy as np
import model_evaluate

import getopt

# hyperparameters
LR = 0.001       #learning rate
TRAINING_ITER = 3000 # 10000   #iteration times
BATCH_SIZE = 18        #batch size of input
 
SEQUENCE_LENGTH = 164   #sequence length of input
EMBEDDING_SIZE = 4      #char embedding size(sequence width of input)
# CONV_SIZE = 3    #first filter size
 # CONV_DEEP = 128   #number of first filter(convolution deepth)
  
STRIDES = [1,1,1,1]  #the strid in each of four dimensions during convolution
KSIZE = [1,164,1,1]    #pooling window size

FC_SIZE = 128     #nodes of full-connection layer
NUM_CLASSES = 2   # classification number
 
DROPOUT_KEEP_PROB = 0.5   #keep probability of dropout

def usage():
    print("USAGE: python create_model_conv6.py --neg neg.csv --pos pos.csv --modelpath ~/modelpath")

def process_argv():
    requireds = ["neg", "pos", "modelpath"]

    try:
        longopts = [ opt + "=" for opt in requireds ]
        opts, args = getopt.getopt(sys.argv[1:], "", longopts)
    except getopt.GetoptError:
        print("Wrong usage!")
        usage()
        sys.exit(1)

    # parse the options
    r = {}
    for op, value in opts:
        if op in ("--neg"):
            r['neg'] = value
        elif op in ("--pos"):
            r['pos'] = value
        elif op in ("--modelpath"):
            r['modelpath'] = value
        elif op in ("-h","--help"):
            usage()
            sys.exit()

    for required in requireds:
        if not required in r:
            print("Wrong usage!!")
            usage()
            sys.exit(1)

    return r

def train_model_conv6(opts):
    log_path = opts['modelpath']
    model_path = log_path + '/model.ckpt'

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    FILE_PATH = opts['neg']
    FILE_PATH_PUTATIVE = opts['pos']

    # FILE_PATH = "datac/ds_canonical.csv"
    # FILE_PATH_PUTATIVE = "datac/ds_mirtrons.csv"

    # FILE_PATH = "../../data/miRBase_set.csv"
    # FILE_PATH_PUTATIVE = "../../data/putative_mirtrons_set.csv"

    all_data_array = dataRead.read_data(FILE_PATH,FILE_PATH_PUTATIVE)
    vectorized_dataset = dataVectorization.vectorize_data(all_data_array, max_seq_len = SEQUENCE_LENGTH)
    # X_train, y_train, X_test, y_test = dataPartition.data_partition(vectorized_dataset)
    X_train, y_train, X_test, y_test = dataPartition.data_partition_2(vectorized_dataset)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    print('y_train.shape', y_train.shape)
    print('X_train.shape', X_train.shape)
    # print('y_test.shape', y_test.shape)
    # print('X_test.shape', X_test.shape)
    print("dataset vectorization finished!")
    print("iteration",TRAINING_ITER)
    dataset_size = int( len(X_train) ) #number of training dataset

    input_X = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH,EMBEDDING_SIZE,1],\
                            name='input_X')
    input_y = tf.placeholder(tf.float32,[None, NUM_CLASSES],name='input_y')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')


    conv6_output = model_construction.model_conv6_output\
                                   (input_X,EMBEDDING_SIZE,keep_prob)
    
    #loss_and_optimization(conv6_output,y_train)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                                (logits = conv6_output,labels = input_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # optimization
    train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy_mean)
    
    # calculate the accuracy of the model
    correct_prediction = tf.equal(tf.argmax(conv6_output,1), tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
    tf.summary.scalar("loss",cross_entropy_mean)
    tf.summary.scalar("accuracy",accuracy)
    # evaluation operation test_data
    tp_op,tn_op,fp_op,fn_op = model_evaluate.evaluation_op(conv6_output,input_y)
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    # strore the operation conv6_output to be used in the prediction(isMirtron.py)
    tf.add_to_collection('pred_network', conv6_output)
    
    # train the model with the training dataset
    
    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # tf.keras.backend.set_session(tf.Session(config=config));

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    # session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Session preparation
    # Forcing tf to use CPU
    sess_config = tf.ConfigProto( device_count = {'GPU': 0} ) 
    # sess_config.gpu_options.allow_growth = True
    # sess = tf.Session(config=sess_config)

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(log_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_ITER):
            start = (i * BATCH_SIZE)% dataset_size
            end = min(start + BATCH_SIZE,dataset_size)
            batch_xs = X_train[start:end]
            batch_ys = y_train[start:end]
            #sess.run(train_op,feed_dict={input_X:batch_xs,input_y:batch_ys,\
            #                              keep_prob:DROPOUT_KEEP_PROB})
            
            _,rs = sess.run([train_op,merged],\
                            feed_dict={input_X:batch_xs,input_y:batch_ys,\
                                      keep_prob:DROPOUT_KEEP_PROB})
            writer.add_summary(rs,i)
            
          #  print loss and accuracy during the training process
            if(i%1000==0):
                print("The {} iteration".format(i))
              #   print("The cross_entropy_mean:",end='')
              #   print(sess.run(cross_entropy_mean,\
              #                  feed_dict={input_X:batch_xs,input_y:batch_ys,\
              #                             keep_prob:DROPOUT_KEEP_PROB}))
              #   #print(sess.run(cross_entropy_mean,\
              #   #               feed_dict={input_X:batch_xs,input_y:batch_ys}))
              # #  print("The accuracy on batch data:",end='')
              # #  print(sess.run(accuracy,feed_dict={input_xs:batch_xs,\
              # #                  input_ys:batch_ys,keep_prob:1}))
              #   print("The accuracy on training data:",end='')
              #   print(sess.run(accuracy,feed_dict={input_X:X_train,\
              #                  input_y:y_train,keep_prob:1}))
              #  # print(sess.run(accuracy,feed_dict={input_X:X_train,\
              #   #               input_y:y_train}))
              #   print("The accuracy on test data:",end='')
              #   print(sess.run(accuracy,feed_dict={input_X:X_test,\
              #                              input_y:y_test,keep_prob:1}))
              #   print("==================")
                
            saver.save(sess,model_path)  
                
        print("*********training finished********")
        # print("performance on the test dataset:")
        # tp,tn,fp,fn = sess.run([tp_op, tn_op,fp_op, fn_op],\
        #                          feed_dict={input_X:X_test,\
        #                          input_y:y_test,keep_prob:1})
        # print("tp:{},tn:{},fp:{},fn:{}".format(tp,tn,fp,fn)) 
        # model_evaluate.print_test_evaluation(tp,tn,fp,fn)

if __name__ == "__main__":
    # This file contains adaptations for automated execution
    # Example:
    # python create_model_conv6.py --canonical datac/ds_canonical.csv --mirtrons datac/ds_mirtrons.csv --modelpath ~/cnnmirtronpred

    opts = process_argv()
    train_model_conv6(opts)
