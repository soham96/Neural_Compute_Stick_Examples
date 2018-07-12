import tensorflow as tf
import numpy as np
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

    #Since the Neural Compute Stick can work with Float 32 values
    #We specify the input format as float32
    x=tf.placeholder(tf.float32, [None, 784], name='input')
    
    conv_output=model(x)

    output = tf.nn.softmax(conv_output, name='output')

    print(f"Saving graph to {model_save_dir}. Please make sure the directory exists!")

    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, os.path.join("resources", "mnist_model", "model"))
        saver.save(sess, model_save_dir)

if __name__ == '__main__':
    
    mnist_dir=os.path.join("resources", "data")
    model_save_dir=os.path.join("resources", "inference_mnist_model", "model")
    main(mnist_dir, model_save_dir)