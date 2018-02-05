# Python 3.6
import tensorflow as tf
import numpy as np
from PIL import Image as pi

VGG16_PARAM_FILE = "../../tensorflow-vgg16-master/vgg16_weights.npz"
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
            self.conv1_2 = tf.nn.bias_add(self.conv1_1, self.conv1_2_b)
            self.conv1_2 = tf.nn.relu(self.conv1_2)

            # add a pooling layer
            self.pool1 = tf.nn.max_pool(
                self.conv1_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1'
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

def main():
    net = ShipNet(VGG16_PARAM_FILE)
    im = pi.open('../../tensorflow-vgg16-master/cat.jpg')
    im = im.resize([IMG_W,IMG_H])
    x = np.array(im, np.float32)
    x = np.reshape(x, [1, IMG_H, IMG_W, IMG_C])
    y = net.run(x)
    im = pi.fromarray(y[0,:,:,1])
    im.show('Pooling#1')
main()
