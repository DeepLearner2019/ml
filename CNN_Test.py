import tensorflow as tf
import ReadData
import numpy as np
import os

import CNN_Model
STEPS = 30000
BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_NAME = 'CNN_ModelFile/model.ckpt'
test_dir = 'test'






x = tf.placeholder(tf.float32, [None, 100,100,3])
y = tf.placeholder(tf.float32, [None, 5])

test_data,test_label=ReadData.get_outorder_data_test(test_dir)



logits = CNN_Model.inference(x,False,None)

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32),tf.cast(tf.argmax(y, 1), tf.int32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
# 启动session
with tf.Session() as sess:
    saver.restore(sess,MODEL_NAME)
    test_acc, n_batch = 0, 0
    for test_data_batch, test_label_batch in ReadData.get_batch(test_data, test_label, BATCH_SIZE):
        ac = sess.run([acc], feed_dict={x: test_data_batch, y: test_label_batch})
        test_acc += ac;
        n_batch += 1
        print("test acc: %f" % (np.sum(test_acc) / n_batch))
