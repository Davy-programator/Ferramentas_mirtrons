"""define the functions to calculate the performance
"""

import tensorflow as tf
import math

def evaluation_op(predic_output,input_ys):
    #calculate TP，TN，FP，FN on test_data
    predictions = tf.argmax(predic_output, 1)
    actuals = tf.argmax(input_ys, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tn_op = tf.reduce_sum(\
        tf.cast(\
            tf.logical_and(\
            tf.equal(actuals, ones_like_actuals),\
            tf.equal(predictions, ones_like_predictions)\
            ), \
            "float")\
        )

    tp_op = tf.reduce_sum(\
        tf.cast(\
            tf.logical_and(\
            tf.equal(actuals, zeros_like_actuals),\
            tf.equal(predictions, zeros_like_predictions)\
            ),\
            "float")\
        )

    fn_op = tf.reduce_sum(\
        tf.cast(\
          tf.logical_and(\
            tf.equal(actuals, zeros_like_actuals),\
            tf.equal(predictions, ones_like_predictions)\
          ),\
          "float")\
        )

    fp_op = tf.reduce_sum(\
        tf.cast(\
          tf.logical_and(\
            tf.equal(actuals, ones_like_actuals),\
            tf.equal(predictions, zeros_like_predictions)\
          ),\
          "float")\
        )
    return tp_op, tn_op,fp_op, fn_op



def print_test_evaluation(tp,tn,fp,fn):  
    tpr = float(tp)/(float(tp) + float(fn))
    recall = tpr
    print("Sensitivity/recall on the test data is :{}".format(tpr)) 

    specifity = float(tn)/(float(tn) + float(fp))
    print("specifity on the test data is :{}".format(specifity)) 

    precision = float(tp)/(float(tp) + float(fp))
    print("precision on the test data is :{}".format(precision))

    f1_score = (2 * (precision * recall)) / (precision + recall)
    print("f1_score on the test data is :{}".format(f1_score))

    fpr = float(fp)/(float(tp) + float(fn))
    print("fpr on the test data is :{}".format(fpr))

    mcc = ((float(tp) * float(tn)) - (float(fp) * float(fn))) /\
            math.sqrt((float(tp) + float(fp)) * (float(tp) + float(fn))*\
             (float(tn) + float(fp)) * (float(tn) + float(fn)))
    print("mcc on the test data is :{}".format(mcc))

    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
    print("accuracy on the test data is :{}".format(accuracy))
