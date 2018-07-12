import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

def model(x):
    '''
    Creates the model and returns it.
    Note: There is no training or preprocessing specific operations here

    RETURNS:
    '''

    x_reshape=tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv_1'):
        initial_weight_conv_1=tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
        initial_bias_conv_1=tf.constant(0.1, shape=[32])

        w_conv_1=tf.Variable(initial_weight_conv_1)
        bias_conv_1=tf.Variable(initial_bias_conv_1)
        conv_layer_1=tf.nn.conv2d(x_reshape, w_conv_1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv_1=tf.nn.relu(conv_layer_1 + bias_conv_1)
    
    with tf.name_scope('conv_2'):
        initial_weight_conv_2=tf.truncated_normal([5, 5, 32, 64], stddev=0.1)
        initial_bias_conv_2=tf.constant(0.1, shape=[64])

        w_conv_2=tf.Variable(initial_weight_conv_2)
        bias_conv_2=tf.Variable(initial_bias_conv_2)
        conv_layer_2=tf.nn.conv2d(h_conv_1, w_conv_2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv_2=tf.nn.relu(conv_layer_2 + bias_conv_2)
    
    # Did NOT apply pooling
    #Read this blog to know more: https://www.saama.com/blog/different-kinds-convolutional-filters/

    with tf.name_scope('fully_connected_1'):
        initial_weight_fc1=tf.truncated_normal([28*28*64, 1024], stddev=0.1)
        initial_bias_fc1=tf.constant(0.1, shape=[1024])

        flatten_fc1=tf.reshape(h_conv_2, [-1, 28*28*64])
        w_fc1=tf.Variable(initial_weight_fc1)
        b_fc1=tf.Variable(initial_bias_fc1)
        h_fc1=tf.nn.relu(tf.matmul(flatten_fc1, w_fc1) + b_fc1)

    with tf.name_scope('fully_connected_2'):
        initial_weight_fc2=tf.truncated_normal([1024, 10], stddev=0.1)
        initial_bias_fc2=tf.constant(0.1, shape=[10])

        w_fc2=tf.Variable(initial_weight_fc2)
        b_fc2=tf.Variable(initial_bias_fc2)

        model_output=tf.matmul(h_fc1, w_fc2)+b_fc2

    return model_output
    





def main(mnist_dir, model_save_dir):
    '''
    Builds, Trains and Saves the Model
    '''

    mnist=input_data.read_data_sets(mnist_dir, one_hot=True)

    #Since the Neural Compute Stick can work with Float 32 values
    #We specify the input format as float32
    x=tf.placeholder(tf.float32, [None, 784], name='input')
    y=tf.placeholder(tf.float32, [None, 10])

    conv_output=model(x)

    with tf.name_scope('loss'):
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=conv_output)
        cross_entropy=tf.reduce_mean(cross_entropy)
    
    with tf.name_scope('optimiser'):
        optimiser=tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    with tf.name_scope('accuracy'):
        correct_pred=tf.equal(tf.argmax(conv_output, 1), tf.argmax(y, 1))
        correct_pred=tf.cast(correct_pred, tf.float32)
    
    accuracy=tf.reduce_mean(correct_pred)

    print(f"Saving graph to {model_save_dir}. Please make sure the directory exists!")

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(500):
            batch=mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy=accuracy.eval(feed_dict={x:batch[0], y:batch[1]})
                print(f"{i}-->{train_accuracy}")
            
            optimiser.run(feed_dict={x:batch[0], y:batch[1]})
        
        # test_accuracy=accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels})
        # print(f"Test Accuracy: {test_accuracy}")

        saver.save(sess, save_path=model_save_dir)


if __name__ == '__main__':
    
    mnist_dir=os.path.join("resources", "data")
    model_save_dir=os.path.join("resources", "mnist_model", "model")
    main(mnist_dir, model_save_dir)