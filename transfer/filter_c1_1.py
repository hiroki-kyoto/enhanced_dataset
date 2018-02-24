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
    return (x-_min)/_dif

def conv(x, w):
    with tf.Session() as sess:
        tf_x = tf.constant(x)
        tf_w = tf.constant(w)
        tf_y = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
        return

def main():
    net = ShipNet(VGG16_PARAM_FILE)
    f = net.conv1_1_W[:, :, :, 52].transpose(2,0,1)
    cond = np.abs(f) < 0.1
    f[cond] = 0

    image = pi.open(ROOT_PATH+'/train/051-Destroyer/qz85.jpg')
    matrix = np.array(image)
    matrix = matrix.transpose([2,0,1])


    image.show()

main()
