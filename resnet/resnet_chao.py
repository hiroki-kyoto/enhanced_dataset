# -*- coding:utf8 -*-
# resnet_test.py
# using well trained resnet tensorflow model
# from $MODEL_PATH to restore a model in
# memory and run test with ship dataset

from resnet.resnet import *
from PIL import Image
import numpy as np

class ResNetLoader:
    def __init__(self, model_path):
        # labels:
        self.labels = [
                'Supply',
                'Destroyer(6607)',
                'Destroyer(051B/C)',
                'Frigate(054A)',
                'Landing(MR)',
                'Frigate(056)',
                'Frigate(054)',
                'Frigate(053K)',
                'Frigate(65)',
                'Destroyer(Sovremenny)',
                'Destroyer(051)',
                'Frigate(053H)',
                'Antisubmarine',
                'Frigate(053H1G)',
                'Mine-Hunter',
                'Missile-Boat',
                'Frigate(053H1Q)',
                'Frigate(6601)',
                'Landing(Hovercraft)',
                'Frigate(053H3)',
                'Frigate(053H2)',
                'Frigate(053H1)',
                'Torpedo-Boat',
                'Destroyer(052)',
                'Landing(S)',
                'Landing(L)',
                'Landing(L071)',
                'Frigate(053H2G)'
        ]
        self.graph = tf.Graph()
        self.ready = False
        with self.graph.as_default():
            # Create a new resnet classifier.
            self.classifier = tf.estimator.Estimator(
                    model_fn = res_net_model,
                    model_dir = model_path
            )
            self.ready = True
        
        self.im_w = W # width
        self.im_h = H # height

    def classify(self, img):
        im = img.resize(
                [self.im_w, self.im_h],
                Image.ANTIALIAS
        )
        im_data = np.array(im, dtype=np.float32)
        assert len(im_data.shape)==3 and im_data.shape[2]==3
        im_data = np.reshape(
                im_data, 
                [1, self.im_h, self.im_w, 3]
        )
        with self.graph.as_default():
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={INPUT_NAME: im_data},
                    y=np.zeros([1]),
                    num_epochs=1,
                    shuffle=False
            )
            res = self.classifier.predict(input_fn = test_input_fn)
            label_id = 0
            for i in res:
                label_id = i['class']
        return label_id

    def classify_with_image_path(self, image_path):
        print(image_path)
        img = Image.open(image_path)
        im = img.resize([self.im_w, self.im_h])
        im_data = np.array(im, dtype=np.float32)
        assert len(im_data.shape)==3 and im_data.shape[2]==3
        im_data = np.reshape(
                im_data,
                [1, im_data.shape[0], im_data.shape[1], im_data.shape[2]]
        )
        with self.graph.as_default():
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={INPUT_NAME: im_data},
                    y=np.zeros([1]),
                    num_epochs=1,
                    shuffle=False
            )
            res = self.classifier.predict(input_fn = test_input_fn)
            label_id = -1
            for i in res:
                label_id = i['class']
                print(i)
        return label_id
    
    def free(self):
        # do nothing
        print('freed')


