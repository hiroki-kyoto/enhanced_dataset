# Using tensorflow 1.4.0
# The models: Alexnet, Resnet

from alexnet import training as alexnet_train
from resnet import resnet_train as resnet_train
from alexnet import alexnet as alexnet_loader
from alexnet import input_data as alexnet_input_data
from resnet import resnet_chao as resnet_loader
import sys
import os
from PIL import Image as pi
import numpy as np

# Training
root_dir = 'E:/code/workspace/enhanced_dataset'
train_dir = root_dir + '/train/'
test_dir = root_dir + '/test/'
alexnet_log_dir = root_dir + '/alexnet_log/'
resnet_log_dir = root_dir + '/resnet_log/'

assert(len(sys.argv)>=3)
_loader = None
if sys.argv[1]=='train':
    if sys.argv[2]=='alexnet':
        alexnet_train.train(train_dir, alexnet_log_dir)
    elif sys.argv[2]=='resnet':
        resnet_train.resnet_train(train_dir, resnet_log_dir)
    else:
        assert(False)
elif sys.argv[1]=='test':
    if sys.argv[2]=='alexnet':
        _loader = alexnet_loader.AlexNetLoader(alexnet_log_dir)
        dirs = os.listdir(test_dir)
        dirs.sort()
        images, labels = alexnet_input_data.get_files(test_dir)
        correct_num = np.zeros(len(dirs))
        expected_num = np.zeros(len(dirs))
        predicted_num = np.zeros(len(dirs))
        for i in range(len(images)):
            pred = _loader.classify_with_im_path(images[i])
            if pred == labels[i]:
                correct_num[pred] += 1
            expected_num[labels[i]] += 1
            predicted_num[pred] += 1
        print("Test results:")
        print("\t%16s\t%8s\t%8s" % ("Category", "Recall", "Precision"))
        for i in range(len(correct_num)):
            expected = max(expected_num[i], 1)
            predicted = max(predicted_num[i], 1)
            correct = correct_num[i]
            print("\t%16s\t%8.2f%%\t%8.2f%%" % (
                dirs[i],
                100.0*correct/expected,
                100.0*correct/predicted))
        print("Overall accuracy:\t%.2f%%" % (100.0*sum(correct_num)/sum(expected_num)))

    elif sys.argv[2]=='resnet':
        _loader = resnet_loader.ResNetLoader(resnet_log_dir)
        dirs = os.listdir(test_dir)
        dirs.sort()
        images = []
        labels = []
        for i, dir in enumerate(dirs):
            subdir = os.path.join(test_dir, dir)
            files = os.listdir(subdir)
            for fn in files:
                images.append(
                    pi.open(os.path.join(subdir, fn))
                )
                labels.append(i)
        correct_num = np.zeros(len(dirs))
        expected_num = np.zeros(len(dirs))
        predicted_num = np.zeros(len(dirs))
        for i in range(len(images)):
            pred = _loader.classify(images[i])
            if pred == labels[i]:
                correct_num[pred] += 1
            expected_num[labels[i]] += 1
            predicted_num[pred] += 1
        print("Test results:")
        print("\t%16s\t%8s\t%8s" % ("Category", "Recall", "Precision"))
        for i in range(len(correct_num)):
            expected = max(expected_num[i], 1)
            predicted = max(predicted_num[i], 1)
            correct = correct_num[i]
            print("\t%16s\t%8.2f%%\t%8.2f%%" % (
                dirs[i],
                100.0 * correct / expected,
                100.0 * correct / predicted))
        print("Overall accuracy:\t%.2f%%" % (100.0 * sum(correct_num) / sum(expected_num)))
    else:
        assert(False)
