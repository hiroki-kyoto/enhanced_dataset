# python 3.6
from ship_net import *
if 'ROOT_PATH' not in globals():
    from transfer.ship_net import  *

from PIL import Image as pi
def main():
    net = ShipNet(VGG16_PARAM_FILE)
    im = pi.open(ROOT_PATH+'/vgg16/filter-analysis/ship/qz85.jpg')
    im = im.resize([IMG_W,IMG_H])
    x = np.array(im, np.float32)
    x = np.reshape(x, [1, IMG_H, IMG_W, IMG_C])
    y = net.run_part(net.x, net.conv3_2, x)
    y = y[0]
    y = np.transpose(y, [2,0,1])
    for filter_id in range(y.shape[0]):
        im = pi.fromarray(y[filter_id])
        im = im.convert('RGB')
        fp = (ROOT_PATH+'/vgg16/filter-analysis/ship/C3_2/%s.jpg') % (filter_id+1)
        im.save(fp)
main()
