#coding=utf-8
import tensorflow as tf


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
    conv1 = tf.nn.bias_add(
        tf.nn.conv2d(
            images,
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


def evaluation(logits, onehot):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.argmax(onehot, axis=1)
        top_1 = tf.nn.in_top_k(logits, labels, 1)
        top_2 = tf.nn.in_top_k(logits, labels, 2)
        top_3 = tf.nn.in_top_k(logits, labels, 3)
        top_1 = tf.reduce_mean(tf.cast(top_1, tf.float16))
        top_2 = tf.reduce_mean(tf.cast(top_2, tf.float16))
        top_3 = tf.reduce_mean(tf.cast(top_3, tf.float16))
        tf.summary.scalar(scope.name + '/top_1', top_1)
        tf.summary.scalar(scope.name + '/top_2', top_2)
        tf.summary.scalar(scope.name + '/top_3', top_3)
    return top_1
