# -*- coding:utf8 -*-
# resnet_test.py
# using well trained resnet tensorflow model
# from ./model/small/ to restore a model in 
# memory and run test with ship dataset

from resnet.resnet import *
from PIL import Image

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
            print(self.classifier)
            self.ready = True
        
        self.im_w = 48 # width
        self.im_h = 32 # height
        self.im_c = 3 # channel

    def classify(self, img):
        im = img.resize(
                [self.im_w, self.im_h], 
                Image.ANTIALIAS
        )
        im_data = np.array(im)
        im_data = np.reshape(
                im_data, 
                [1, self.im_h, self.im_w, self.im_c]
        )
        im_data = im_data.astype(np.float32)
        im_data = np.multiply(im_data, 1.0/255.0)
        
        with self.graph.as_default():
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={X_FEATURE: im_data},
                    y=np.zeros([1]),
                    num_epochs=1,
                    shuffle=False
            )
            res = self.classifier.predict(input_fn = test_input_fn)
            label_id = 0
            prob = 0.0
            for i in res:
                label_id = i['class']
        #return self.labels[label_id]
        return label_id
    
    def free(self):
        # do nothing
        print('freed')


