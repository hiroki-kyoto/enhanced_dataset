#coding=utf-8
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
  
def get_batches_with_onehot(
        images,
        labels,
        resize_w,
        resize_h,
        batch_size,
        capacity,
        N_CLASS
):
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    onehots = tf.one_hot(labels, N_CLASS, on_value=1.0, off_value=0.0)
    queues = tf.train.slice_input_producer([images, onehots])
    labels = queues[1]
    images_binary = tf.read_file(queues[0])
    images_decoded = tf.image.decode_jpeg(
        images_binary,
        channels=3
    )
    images = tf.image.resize_image_with_crop_or_pad(
        images_decoded,
        resize_w,
        resize_h
    )
    images = tf.image.per_image_standardization(images)
    image_batches, label_batches = tf.train.batch(
        [images, labels],
        batch_size=batch_size,
        num_threads=64,
        capacity=capacity
    )
    image_batches = tf.cast(image_batches, tf.float32)
    label_batches = tf.reshape(label_batches, [batch_size, N_CLASS])
    return image_batches, label_batches