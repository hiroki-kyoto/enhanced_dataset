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
                tf_y,
                ksize=[1,3,3,1],
                strides=[1,1,1,1],
                padding='SAME'
            )
            y = sess.run(tf_y)
            y = y[0,:,:,0]
            return y

def sparse(x, threshold):
    cond = x < threshold
    x[cond] = 0.0
    return x

def main():
    net = ShipNet(VGG16_PARAM_FILE)
    w = net.conv1_1_W[:, :, :, [7,46,52]]
    w = np.transpose(w, [3,0,1,2])
    b = net.conv1_1_b[[7,46,52]]
    cond = np.abs(w) < 0.1
    w[cond] = 0
    image = pi.open(ROOT_PATH+'/ship-detection/train/051-Destroyer/qz96.jpg')
    image = image.resize([320, 320])
    im_data = np.array(image, dtype=np.float32)
    x = im_data - RGB_MEAN
    y1 = f(x, w[0], b[0])
    y2 = f(x, w[1], b[1])
    y3 = f(x, w[2], b[2])
    y = np.minimum(y1, y2)
    y = np.minimum(y, y3)
    cond = (y < 0.1)
    im_data[cond] = 0
    im_data = im_data.astype(np.uint8)
    im = pi.fromarray(im_data, mode='RGB')
    im.show()



main()
