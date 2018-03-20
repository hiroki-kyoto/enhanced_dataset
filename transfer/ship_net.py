# Python 3.6
import tensorflow as tf
import numpy as np

ROOT_PATH = 'C:/Users/work/Desktop/shigoto'
VGG16_PARAM_FILE = ROOT_PATH + "/tensorflow-vgg16-master/vgg16_weights.npz"
IMG_N = 1
IMG_H = 224
IMG_W = 224
IMG_C = 3
RGB_MEAN = [123.68,116.779,103.939]

class ShipNet(object):
    def __init__(self, vgg_param_file):
        # graph and session
        self.graph = tf.Graph()
        self.sess = None
        self.saver = None
        with self.graph.as_default():
            # load vgg filter weights from a local file
            self.vgg_param = np.load(vgg_param_file)
            self.sess = tf.Session()
            # the first convolution layer param
            self.conv1_1_W = self.vgg_param['conv1_1_W']
            self.conv1_1_b = self.vgg_param['conv1_1_b']
            self.x = tf.placeholder(
                tf.float32,
                shape=[IMG_N, IMG_H, IMG_W, IMG_C]
            )
            assert(IMG_C==3)
            self.x_mean = tf.constant(RGB_MEAN, dtype=tf.float32)
            self.x_mean = tf.reshape(self.x_mean, [1,1,1,3])
            self.x_std = self.x - self.x_mean
            self.conv1_1 = tf.nn.conv2d(
                self.x_std,
                self.conv1_1_W,
                [1,1,1,1],
                padding='SAME'
            )
            self.conv1_1 = tf.nn.bias_add(self.conv1_1, self.conv1_1_b)
            self.conv1_1 = tf.nn.relu(self.conv1_1)

            # the second convolution layer
            self.conv1_2_W = self.vgg_param['conv1_2_W']
            self.conv1_2_b = self.vgg_param['conv1_2_b']
            self.conv1_2 = tf.nn.conv2d(
                self.conv1_1,
                self.conv1_2_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv1_2 = tf.nn.bias_add(self.conv1_2, self.conv1_2_b)
            self.conv1_2 = tf.nn.relu(self.conv1_2)

            # add a pooling layer
            self.pool1 = tf.nn.max_pool(
                self.conv1_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding='SAME',
            )
            self.pool1 = self.conv1_2

            # second group of convolution layers
            self.conv2_1_W = self.vgg_param['conv2_1_W']
            self.conv2_1_b = self.vgg_param['conv2_1_b']
            self.conv2_1 = tf.nn.conv2d(
                self.pool1,
                self.conv2_1_W,
                [1,1,1,1],
                padding='SAME'
            )
            self.conv2_1 = tf.nn.bias_add(self.conv2_1, self.conv2_1_b)
            self.conv2_1 = tf.nn.relu(self.conv2_1)

            self.conv2_2_W = self.vgg_param['conv2_2_W']
            self.conv2_2_b = self.vgg_param['conv2_2_b']
            self.conv2_2 = tf.nn.conv2d(
                self.conv2_1,
                self.conv2_2_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv2_2 = tf.nn.bias_add(self.conv2_2, self.conv2_2_b)
            self.conv2_2 = tf.nn.relu(self.conv2_2)

            self.pool2 = tf.nn.max_pool(
                self.conv2_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            self.pool2 = self.conv2_2

            # the third convolutional layers
            self.conv3_1_W = self.vgg_param['conv3_1_W']
            self.conv3_1_b = self.vgg_param['conv3_1_b']
            self.conv3_1 = tf.nn.conv2d(
                self.pool2,
                self.conv3_1_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv3_1 = tf.nn.bias_add(self.conv3_1, self.conv3_1_b)
            self.conv3_1 = tf.nn.relu(self.conv3_1)

            self.conv3_2_W = self.vgg_param['conv3_2_W']
            self.conv3_2_b = self.vgg_param['conv3_2_b']
            self.conv3_2 = tf.nn.conv2d(
                self.conv3_1,
                self.conv3_2_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv3_2 = tf.nn.bias_add(self.conv3_2, self.conv3_2_b)
            self.conv3_2 = tf.nn.relu(self.conv3_2)

            self.conv3_3_W = self.vgg_param['conv3_3_W']
            self.conv3_3_b = self.vgg_param['conv3_3_b']
            self.conv3_3 = tf.nn.conv2d(
                self.conv3_2,
                self.conv3_3_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv3_3 = tf.nn.bias_add(self.conv3_3, self.conv3_3_b)
            self.conv3_3 = tf.nn.relu(self.conv3_3)

            self.pool3 = tf.nn.max_pool(
                self.conv3_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding='SAME'
            )

            # the fourth group
            self.conv4_1_W = self.vgg_param['conv4_1_W']
            self.conv4_1_b = self.vgg_param['conv4_1_b']
            self.conv4_1 = tf.nn.conv2d(
                self.pool3,
                self.conv4_1_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv4_1 = tf.nn.bias_add(self.conv4_1, self.conv4_1_b)
            self.conv4_1 = tf.nn.relu(self.conv4_1)

            self.conv4_2_W = self.vgg_param['conv4_2_W']
            self.conv4_2_b = self.vgg_param['conv4_2_b']
            self.conv4_2 = tf.nn.conv2d(
                self.conv4_1,
                self.conv4_2_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv4_2 = tf.nn.bias_add(self.conv4_2, self.conv4_2_b)
            self.conv4_2 = tf.nn.relu(self.conv4_2)

            self.conv4_3_W = self.vgg_param['conv4_3_W']
            self.conv4_3_b = self.vgg_param['conv4_3_b']
            self.conv4_3 = tf.nn.conv2d(
                self.conv4_2,
                self.conv4_3_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv4_3 = tf.nn.bias_add(self.conv4_3, self.conv4_3_b)
            self.conv4_3 = tf.nn.relu(self.conv4_3)

            self.pool4 = tf.nn.max_pool(
                self.conv4_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding='SAME'
            )

            # the fifth group of convolution layers
            self.conv5_1_W = self.vgg_param['conv5_1_W']
            self.conv5_1_b = self.vgg_param['conv5_1_b']
            self.conv5_1 = tf.nn.conv2d(
                self.pool4,
                self.conv5_1_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv5_1 = tf.nn.bias_add(self.conv5_1, self.conv5_1_b)
            self.conv5_1 = tf.nn.relu(self.conv5_1)

            self.conv5_2_W = self.vgg_param['conv5_2_W']
            self.conv5_2_b = self.vgg_param['conv5_2_b']
            self.conv5_2 = tf.nn.conv2d(
                self.conv5_1,
                self.conv5_2_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv5_2 = tf.nn.bias_add(self.conv5_2, self.conv5_2_b)
            self.conv5_2 = tf.nn.relu(self.conv5_2)

            self.conv5_3_W = self.vgg_param['conv5_3_W']
            self.conv5_3_b = self.vgg_param['conv5_3_b']
            self.conv5_3 = tf.nn.conv2d(
                self.conv5_2,
                self.conv5_3_W,
                [1, 1, 1, 1],
                padding='SAME'
            )
            self.conv5_3 = tf.nn.bias_add(self.conv5_3, self.conv5_3_b)
            self.conv5_3 = tf.nn.relu(self.conv5_3)

            self.pool5 = tf.nn.max_pool(
                self.conv5_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding='SAME'
            )

            #self.saver = tf.train.Saver()
            #self.sess.run(tf.global_variables_initializer())

    def run(self, x):
        with self.graph.as_default():
            return self.sess.run(self.pool1, feed_dict={self.x:x})

    def run_part(self, tf_node_start, tf_node_end, np_feed_in):
        with self.graph.as_default():
            return self.sess.run(
                tf_node_end,
                feed_dict={tf_node_start:np_feed_in}
            )
