#coding=utf-8
import os
import numpy as np
import tensorflow as tf
import alexnet.input_data as input_data
import alexnet.model as model
import matplotlib.pyplot as plt
from PIL import Image

Labels = {
    '0': (0, 'huweijian'),
    '1': (1, 'huweijian'),
    '2': (2, 'huweijian'),
    '3': (3, 'huweijian'),
    '4': (4, 'huweijian'),
    '5': (4, 'huweijian'),
}

batch_size = 72

def num2class(n):
    x = Labels.items()
    for name, item in x:
        if n in item:
            return name

def get_one_image(img_dir):
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([227, 227])
    image = np.array(image)
    return image

def evaluate_one_image(image_array, logs_train_dir):
    with tf.Graph().as_default():
        N_CLASSES = 6
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 227, 227, 3])
        logit = model.inference4train(image, N_CLASSES)
        logit = tf.nn.softmax(logit)
        x = tf.placeholder(tf.float32, shape=[227, 227, 3])
        # you need to change the directories to yours.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            name = num2class(max_index)
            print('this is a %s with possibility %.6f' % (name, prediction[:, max_index]))

# test model
def test(logs_train_dir, test_dir):
    N_CLASS=6
    train, train_label = input_data.get_files(test_dir)
    train_batch,train_label_batch=input_data.get_batches(train,
                                    train_label,
                                    227,
                                    227,
                                    batch_size,
                                    batch_size)
    train_logits = model.inference4train(train_batch, N_CLASS)
    train__acc = model.evaluation(train_logits, train_label_batch)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    #test
    try:
        for step in np.arange(1):
            if coord.should_stop():
                    break
            tra_acc = sess.run(train__acc)
            print('Step %d, Every %d images, train accuracy = %.2f%%' %(step, batch_size, tra_acc))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
