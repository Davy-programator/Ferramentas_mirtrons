"""Train the constructed cnn model with the training dataset
"""

import sys
sys.path.append("../modelConstruct")
sys.path.append("../dataPreprocess")
sys.path.append("../modelEvaluate")
#from model_construction import *
import dataRead
import dataVectorization
import dataPartition
import model_construction
import tensorflow as tf
import numpy as np
import model_evaluate
import math

# hyperparameters
LR = 0.001       #learning rate
TRAINING_ITER = 10000  #iteration times
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

FILE_PATH = "../../data/miRBase_set.csv"
FILE_PATH_PUTATIVE = "../../data/putative_mirtrons_set.csv"
all_data_array = dataRead.read_data(FILE_PATH,FILE_PATH_PUTATIVE)
vectorized_dataset = dataVectorization.vectorize_data(all_data_array)
X_train, y_train, X_test, y_test = dataPartition.data_partition\
                                                         (vectorized_dataset)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(y_train.shape)
print(X_train.shape)
print(y_test.shape)
print(X_test.shape)
print("dataset vectorization finished!")
print("iteration",TRAINING_ITER)
dataset_size = len(X_train)  #number of training dataset

input_X = tf.placeholder(tf.float32,[None,SEQUENCE_LENGTH,EMBEDDING_SIZE,1])
input_y = tf.placeholder(tf.float32,[None, NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32)

def train_model_conv3(log_path,model_path):
    conv3_output = model_construction.model_conv3_output\
                                   (input_X,EMBEDDING_SIZE,keep_prob)
    
    #loss_and_optimization(conv3_output,y_train)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                                (logits = conv3_output,labels = input_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # optimization
    train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy_mean)  
    
    # calculate the accuracy of the model
    correct_prediction = tf.equal(tf.argmax(conv3_output,1), tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
    tf.summary.scalar("loss",cross_entropy_mean)
    tf.summary.scalar("accuracy",accuracy)
    # evaluation operation test_data
    tp_op,tn_op,fp_op,fn_op = model_evaluate.evaluation_op(conv3_output,input_y)
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    
    # train the model with the training dataset
    with tf.Session() as sess:  # run the session        
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
                print("The {} iteration:".format(i))
                print("The cross_entropy_mean为：",end='')
                print(sess.run(cross_entropy_mean,\
                               feed_dict={input_X:batch_xs,input_y:batch_ys,\
                                          keep_prob:DROPOUT_KEEP_PROB}))
                #print(sess.run(cross_entropy_mean,\
                #               feed_dict={input_X:batch_xs,input_y:batch_ys}))
              #  print("The accuracy on batch data:",end='')
              #  print(sess.run(accuracy,feed_dict={input_xs:batch_xs,\
              #                  input_ys:batch_ys,keep_prob:1}))
                print("The accuracy on training data:",end='')
                print(sess.run(accuracy,feed_dict={input_X:X_train,\
                               input_y:y_train,keep_prob:1}))
               # print(sess.run(accuracy,feed_dict={input_X:X_train,\
                #               input_y:y_train}))
                print("The accuracy on test data:",end='')
                print(sess.run(accuracy,feed_dict={input_X:X_test,\
                                           input_y:y_test,keep_prob:1}))
                print("==================")
                
            saver.save(sess,model_path)  
                
        print("*********training finished********")

        print("performance on the test dataset:")
        tp,tn,fp,fn = sess.run([tp_op, tn_op,fp_op, fn_op],\
                                feed_dict={input_X:X_test,\
                                input_y:y_test,keep_prob:1})
        print("tp:{},tn:{},fp:{},fn:{}".format(tp,tn,fp,fn))
    
        model_evaluate.print_test_evaluation(tp,tn,fp,fn)

if __name__ == "__main__":
    train_model_conv3("../../logs/filter3","../../logs/filter3/filter3.ckpt")

