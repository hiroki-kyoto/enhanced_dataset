#coding=utf-8
import tensorflow as tf
import numpy as np

RGB_MEAN = [123.68,116.779,103.939]

# VGG-16 model file path
ROOT_PATH = 'C:/Users/work/Desktop/shigoto/'
VGG16_PARAM_FILE = ROOT_PATH + "/tensorflow-vgg16-master/vgg16_weights.npz"

# min-max normalization
# x is required to be a 4D tensor with data-format of 'NHWC'
def min_max_normalize(x):
    # transform from NHWC format to NCHW format
    _max = tf.reduce_max(x, axis=[1,2])
    _min = tf.reduce_min(x, axis=[1,2])
    _max = tf.reshape(_max, [_max.shape[0], 1, 1, _max.shape[1]])
    _min = tf.reshape(_min, [_min.shape[0], 1, 1, _min.shape[1]])
    _dif = _max - _min
    _dif = tf.maximum(_dif, 1.0)
    x = tf.minimum((x-_min)/_dif, 255.0/256.0)
    return x

# to get sparse maps
def sparse(x, threshold):
    mask = x > threshold
    mask = tf.cast(mask, tf.float32)
    return mask

# to apply the sparsity mask
def apply_mask(x, mask):
    return x * mask

# using VGG-16 filters
def preprocess(images):
    x = images - RGB_MEAN
    param = np.load(VGG16_PARAM_FILE)
    conv1_1_W = param['conv1_1_W']
    conv1_1_b = param['conv1_1_b']
    w = conv1_1_W[:, :, :, [7, 46, 52]]
    b = conv1_1_b[[7, 46, 52]]
    cond = np.abs(w) < 0.1
    w[cond] = 0
    preprocess_w = tf.get_variable(
        name='preprocess_weights',
        initializer=w,
        trainable=False
    )
    preprocess_b = tf.get_variable(
        name='preprocess_bias',
        initializer=b,
        trainable=False
    )
    # convolve the images
    maps = tf.nn.conv2d(
        input=x,
        filter=preprocess_w,
        strides=[1,1,1,1],
        padding='SAME',
        use_cudnn_on_gpu=False,
        data_format='NHWC',
        name='preprocess_conv2d'
    )
    maps = tf.nn.bias_add(
        value=maps,
        bias=preprocess_b,
        data_format='NHWC',
        name='preprocess_conv2d'
    )
    # nonlinear transform
    maps = tf.nn.relu(
        features=maps
    )
    # normalize the maps
    maps = min_max_normalize(maps)
    maps = tf.nn.max_pool(
        maps,
        ksize=[1,3,3,1],
        strides=[1,1,1,1],
        padding='SAME'
    )
    # syntheses all filters as noise reduction module
    maps = tf.reduce_min(
        input_tensor=maps,
        axis=[3]
    )
    # get sparse masks of maps
    masks = sparse(maps, 0.1)
    # require a shape match
    masks = tf.reshape(
        masks,
        [masks.shape[0],masks.shape[1],masks.shape[2],1]
    )
    maps = apply_mask(images, masks)

    return maps



def inference4train(train_batch, n_classes):
    with tf.variable_scope("weights"):
        weights = {
            # 39*39*3->36*36*20->18*18*20
            'conv1': tf.get_variable(
                'conv1',
                [11, 11, 3, 16],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            # 18*18*20->16*16*40->8*8*40
            'conv2': tf.get_variable(
                'conv2',
                [5, 5, 16, 16],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            # 8*8*40->6*6*60->3*3*60
            'conv3': tf.get_variable(
                'conv3',
                [3, 3, 16, 2],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            # 3*3*60->120
            'conv4': tf.get_variable(
                'conv4',
                [3, 3, 2, 2],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            'conv5': tf.get_variable(
                'conv5',
                [3, 3, 2, 2],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            'fc1': tf.get_variable(
                'fc1',
                [6 * 6 * 2, 10],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            'fc2': tf.get_variable(
                'fc2',
                [10, 10],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
            # 120->6
            'fc3': tf.get_variable(
                'fc3',
                [10, n_classes],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1, dtype=tf.float32)
            ),
        }
    with tf.variable_scope("biases"):
        biases = {
            'conv1': tf.get_variable(
                'conv1',
                [16, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'conv2': tf.get_variable(
                'conv2',
                [16, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'conv3': tf.get_variable(
                'conv3',
                [2, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'conv4': tf.get_variable(
                'conv4',
                [2, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'conv5': tf.get_variable(
                'conv5',
                [2, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'fc1': tf.get_variable(
                'fc1',
                [10, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'fc2': tf.get_variable(
                'fc2',
                [10, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            ),
            'fc3': tf.get_variable(
                'fc3',
                [n_classes, ],
                initializer=tf.constant_initializer(
                    value=0.0, dtype=tf.float32)
            )
        }

    images = tf.reshape(train_batch, shape=[-1, 227, 227, 3])
    images = tf.cast(images, tf.float32)
    # add preprocessing module using trained filters of VGG16
    maps = preprocess(images)
    #maps = tf.cast(maps, tf.uint8)
    #return maps
    maps = maps - RGB_MEAN
    conv1 = tf.nn.bias_add(
        tf.nn.conv2d(
            maps,
            weights['conv1'],
            strides=[1, 4, 4, 1],
            padding='VALID'
        ),
        biases['conv1'])
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(
        relu1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )
    pool1 = tf.nn.lrn(
        pool1,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75,
        name='norm1'
    )
    conv2 = tf.nn.bias_add(
        tf.nn.conv2d(
            pool1,
            weights['conv2'],
            strides=[1, 1, 1, 1],
            padding='SAME'
        ),
        biases['conv2']
    )
    relu2 = tf.nn.relu(conv2)
    
    relu2 = tf.nn.lrn(
        relu2,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75,
        name='norm2'
    )
    pool2 = tf.nn.max_pool(
        relu2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )
    conv3 = tf.nn.bias_add(
        tf.nn.conv2d(
            pool2,
            weights['conv3'],
            strides=[1, 1, 1, 1],
            padding='SAME'
        ),
        biases['conv3']
    )
    relu3 = tf.nn.relu(conv3)
    conv4 = tf.nn.bias_add(
        tf.nn.conv2d(
            relu3,
            weights['conv4'],
            strides=[1, 1, 1, 1],
            padding='SAME'
        ),
        biases['conv4']
    )
    conv5 = tf.nn.bias_add(
        tf.nn.conv2d(
            conv4,
            weights['conv5'],
            strides=[1, 1, 1, 1],
            padding='SAME'
        ),
        biases['conv5']
    )
    relu5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(
        relu5,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )
    pool5 = tf.nn.lrn(
        pool5,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75,
        name='norm2'
    )
    flatten = tf.reshape(
        pool5,
        [-1, weights['fc1'].get_shape().as_list()[0]]
    )
    drop1 = tf.nn.dropout(flatten, 0.5)
    fc1 = tf.matmul(drop1, weights['fc1']) + biases['fc1']
    fc_relu1 = tf.nn.relu(fc1)
    fc2 = tf.matmul(fc_relu1, weights['fc2']) + biases['fc2']
    fc_relu2 = tf.nn.relu(fc2)
    fc3 = tf.matmul(fc_relu2, weights['fc3']) + biases['fc3']
    #fc_relu3 = tf.nn.relu(fc3)
    return fc3


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def losses_with_onehot(logits, onehot):
    with tf.variable_scope('loss') as scope:
        logits_max = tf.reduce_max(logits, axis=[1])
        logits_max = tf.reshape(logits_max, [logits_max.shape[0], 1])
        logits_min = tf.reduce_min(logits, axis=[1])
        logits_min = tf.reshape(logits_min, [logits_min.shape[0], 1])
        logits_dif = logits_max - logits_min
        logits_dif = tf.maximum(logits_dif, 1.0)
        logits_std = (logits - logits_min) / logits_dif
        err = logits_std - onehot
        err = tf.reduce_sum(tf.square(err), axis=[1])
        loss = tf.reduce_mean(err, name='loss-mean')
        tf.summary.scalar(scope.name + '/mean-loss', loss)
        return loss

def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation_with_onehot(logits, onehot):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.argmax(onehot, axis=1)
        top_1 = tf.nn.in_top_k(logits, labels, 1)
        top_1 = tf.reduce_mean(tf.cast(top_1, tf.float16))
        tf.summary.scalar(scope.name + '/top_1', top_1)
    return top_1


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        top_1 = tf.nn.in_top_k(logits, labels, 1)
        top_1 = tf.reduce_mean(tf.cast(top_1, tf.float16))
        tf.summary.scalar(scope.name + '/top_1', top_1)
    return top_1