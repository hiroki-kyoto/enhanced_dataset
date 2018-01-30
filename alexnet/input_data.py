#coding=utf-8
import numpy as np
import tensorflow as tf
import os

def get_files(filename):
    images_train = []
    labels_train = []
    dirs = os.listdir(filename)
    dirs.sort()
    for i in range(len(dirs)):
        class_path = os.path.join(filename, dirs[i])
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            images_train.append(image_path)
            labels_train.append(i)
    return (images_train, labels_train)

def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size,
                                              num_threads = 64, capacity = capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, labels_batch
  
