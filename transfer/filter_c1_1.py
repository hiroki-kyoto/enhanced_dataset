# the 52th filter analysis
from ship_net import *
from PIL import Image as pi

if 'ROOT_PATH' not in globals():
    from transfer.ship_net import *

def preprocess(x):
    return x - RGB_MEAN

def min_max_normalize(x):
    _max = tf.reduce_max(x)
    _min = tf.reduce_min(x)
    _dif = _max - _min
    _dif = tf.maximum(_dif, 1.0)
    return tf.minimum((x-_min)/_dif, 255/256)

def f(x, w, b):
    w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1)
    x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
    b = np.array([b])

    with tf.Session() as sess:
        tf_x = tf.constant(x)
        tf_x = preprocess(tf_x)
        tf_w = tf.constant(w)
        tf_b = tf.constant(b)
        tf_y = tf.nn.conv2d(tf_x, tf_w, strides=[1,1,1,1], padding='SAME')
        tf_y = tf.nn.bias_add(tf_y, tf_b)
        tf_y = tf.nn.relu(tf_y)

        y = sess.run(tf_y)
        y = y[0,:,:,0]
        return y

def main():
    net = ShipNet(VGG16_PARAM_FILE)
    w = net.conv1_1_W[:, :, :, 52]
    b = net.conv1_1_b[52]
    #w = w.transpose([2,0,1])
    cond = np.abs(w) < 0.1
    w[cond] = 0

    image = pi.open(ROOT_PATH+'/train/051-Destroyer/qz86.jpg')
    x = np.array(image, dtype=np.float32)
    #x = x.transpose([2,0,1])

    y = f(x, w, b)
    print(np.max(y))
    print(y.shape)
    im = pi.fromarray(y*256)
    im.show()

main()
