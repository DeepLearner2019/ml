import tensorflow as tf
import ReadData
import numpy as np
import os
STEPS = 30000
BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = 'CNN_ModelFile/'
MODEL_NAME = 'model.ckpt'
train_dir = 'train'
test_dir = 'test'

def conv_op(input_op, name, kh, kw,n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    # 也就是说，它的主要目的是为了更加方便地管理参数命名。
    # 与 tf.Variable() 结合使用。简化了命名
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[kh, kw, n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases= tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p +=[kernel, biases]
        return activation

def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out]), dtype=tf.float32, name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
     return tf.nn.max_pool(
                         input_op,
                         ksize=[1, kh, kw, 1],
                         strides=[1, dh, dw, 1],
                         padding='SAME',
                         name=name)

def VGGNet_11(input_op, keep_prob):
    p = []
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    #conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_1, name="pool1", kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    #conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_1, name="pool2", kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_2, name="pool3", kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_2, name="pool4", kh=2, kw=2, dw=2, dh=2)

    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_2, name="pool5", kh=2, kw=2, dw=2, dh=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")
    fc6 = fc_op(resh1, name="fc6", n_out=2048, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    fc8 = fc_op(fc7_drop, name="fc8", n_out=5, p=p)

    return fc8


def train():

    x = tf.placeholder(tf.float32, [None, 224,224,3])
    y = tf.placeholder(tf.float32, [None, 5])

    train_data,train_label=ReadData.get_outorder_data_train(train_dir)
    test_data, test_label = ReadData.get_outorder_data_test(test_dir)

    #logits = inference(x,True,regularizer)
    #logits = alexnet(x, 0.5, 5)
    #logits = vgg_net(x)
    logits = VGGNet_11(x, 0.5)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(loss, name='loss')

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.cast(tf.argmax(y, 1), tf.int32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    # Tensorboard
    filewriter_path = 'tensorboard'
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    merged_summary = tf.summary.merge_all()
    #merge_summary = tf.summary.merge([loss_summary, acc_summary])
    writer = tf.summary.FileWriter(filewriter_path)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    f=open("loss_acc.txt","a+")
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       #saver.restore(sess,  MODEL_SAVE_PATH+MODEL_NAME)
       for i in range(60):  # 20000
           train_loss, train_acc, n_batch = 0, 0, 0
           for train_data_batch, train_label_batch in ReadData.get_batch(train_data, train_label, BATCH_SIZE):
               _, err= sess.run([train_op, loss], feed_dict={x: train_data_batch, y: train_label_batch})
               train_loss += err
               n_batch += 1
           print(i," train loss: %f" % (np.sum(train_loss) / n_batch))

           saver.save(sess, MODEL_SAVE_PATH+MODEL_NAME)
           test_acc= 0
           test_acc = sess.run(acc, feed_dict={x: test_data, y: test_label})
           print(i," test acc: %f" % test_acc)
           new_context = str(i)+" " +str((np.sum(train_loss) / n_batch))+" " +str(test_acc)+ '\n'
           f.write(new_context)
           result = sess.run(merged_summary, feed_dict={x: test_data, y: test_label})
           writer.add_summary(result, i)
       writer.close()
       f.close()


if __name__ == '__main__':
    train()