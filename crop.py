# Using tensorflow 1.4.0
# The models: SSD

from SSD import ssd
import os
from PIL import Image as pi

# Training
root_dir = 'C:/Users/work/Desktop/shigoto/ship-detection'
input_dir = root_dir + '/crop_input'
output_dir = root_dir + '/crop_output'
ssd_model_dir = root_dir + '/checkpoints/ssd_300_vgg.ckpt'

# check dirs
assert os.path.isdir(input_dir)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
assert os.path.isdir(output_dir)

_target = 'boat'
_detector = ssd.SSDLoader(ssd_model_dir)
if _detector.ready:
    dirs = os.listdir(input_dir)
    dirs.sort()
    for i, dir in enumerate(dirs):
        subdir = os.path.join(input_dir, dir)
        output_subdir = os.path.join(output_dir, dir)
        if not os.path.isdir(output_subdir):
            os.mkdir(output_subdir)
        files = os.listdir(subdir)
        for fn in files:
            objs, bboxes = _detector.detect_with_image_path(
                os.path.join(subdir, fn)
            )
            im = pi.open(os.path.join(subdir, fn))
            w, h = im.size
            # crop the image with bounding boxes
            for i in range(len(objs)):
                if objs[i]==_target:
                    x1, x2 = int(w*bboxes[i][1]), int(w*bboxes[i][3])
                    y1, y2 = int(h*bboxes[i][0]), int(h*bboxes[i][2])
                    im_cropped = im.crop([x1, y1, x2, y2])
                    im_cropped.save(os.path.join(output_subdir, fn))


