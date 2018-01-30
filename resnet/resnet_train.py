# resnet_train.py
# training resnet with ship dataset
# training models are stored in ./model/
# training epoch number <= 1 million

from resnet.resnet import *
import numpy as np
import os
from PIL import Image as pi

# normalization is applied for each image
def load_images_and_labels(dir):
    images = []
    labels = []
    subdirs = os.listdir(dir)
    subdirs.sort()
    for i in range(len(subdirs)):
        class_path = os.path.join(dir, subdirs[i])
        files = os.listdir(class_path)
        for image in files:
            image_path = os.path.join(class_path, image)
            # convert the image into HWC format
            im = pi.open(image_path)
            im = im.resize([W,H])
            im_arr = np.array(im, dtype=np.float32)
            assert(len(im_arr.shape)==3)
            assert(im_arr.shape[2]==3)
            images.append(im_arr)
            labels.append(i)
    return np.array(images), np.array(labels)

def resnet_train(data_dir, log_dir):
    # load data
    images, labels = load_images_and_labels(data_dir)
    # create a new resnet classifier
    classifier = tf.estimator.Estimator(
        model_fn=res_net_model,
        model_dir=log_dir
    )
    # turn on logging system
    tf.logging.set_verbosity(tf.logging.INFO)
    # training input function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {'x' : images},
            y = labels.astype(np.int32),
            batch_size = 8,
            num_epochs = None, # not defined here
            shuffle = True
    )
    classifier.train(input_fn = train_input_fn, steps=20000)
    print('training done.')
