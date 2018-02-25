# the 52th filter analysis
from ship_net import *
from PIL import Image as pi

if 'ROOT_PATH' not in globals():
    from transfer.ship_net import *

def min_max_normalize(x):
    _max = tf.reduce_max(x)
    _min = tf.reduce_min(x)
    _dif = _max - _min
    _dif = tf.maximum(_dif, 1.0)
    return tf.minimum((x-_min)/_dif, 255/256)

def f(x, w, b):
    w = w.reshape([w.shape[0], w.shape[1], w.shape[2], 1])
    x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
    b = np.array([b])
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            tf_x = tf.constant(x, dtype=tf.float32)
            tf_w = tf.constant(w, dtype=tf.float32)
            tf_b = tf.constant(b, dtype=tf.float32)
            tf_y = tf.nn.conv2d(
                tf_x,
                tf_w,
                strides=[1,1,1,1],
                padding='SAME'
            )
            tf_y = tf.nn.bias_add(tf_y, tf_b)
            tf_y = tf.nn.relu(tf_y)
            # a max pooling
            tf_y = min_max_normalize(tf_y)
            tf_y = tf.nn.max_pool(
                tf_y, ksize=[1,3,3,1],
                strides=[1,1,1,1],
                padding='SAME'
            )
            y = sess.run(tf_y)
            y = y[0,:,:,0]
            return y

def main():
    net = ShipNet(VGG16_PARAM_FILE)
    w = net.conv1_1_W[:, :, :, 52]
    b = net.conv1_1_b[52]
    cond = np.abs(w) < 0.1
    w[cond] = 0

    image = pi.open(ROOT_PATH+'/train/051-Destroyer/qz88.jpg')
    x = np.array(image, dtype=np.float32)
    x = x - RGB_MEAN
    y = f(x, w, b)*256
    im = pi.fromarray(y)
    im.show()
    # apply filter again
    w_single = np.mean(w, axis=(2)).astype(np.float32)
    w_single = w_single.reshape(w_single.shape[0], w_single.shape[1], 1)
    x_single = y - np.mean(np.array(RGB_MEAN)).astype(np.float32)
    x_single = x_single.reshape(x_single.shape[0], x_single.shape[1], 1)
    print(x_single.shape)
    y1 = f(x_single, w_single, b) * 256

    im1 = pi.fromarray(y1)
    im1.show()


main()
