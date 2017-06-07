import numpy as np
import chainer
from chainer.links import caffe

class CaffeNet(caffe.CaffeFunction):
    def __init__(self, model_path):
        super(CaffeNet, self).__init__(model_path)

    def preprocess(self, image):
        return np.rollaxis(image[:,:,::-1] - self.mean, 2)

    def deprocess(self, image):
        return (np.rollaxis(image, 0, image.ndim) + self.mean)[:,:,::-1]

class GoogleNet(CaffeNet):
    def __init__(self, model_path):
        super(GoogleNet, self).__init__(model_path)
        self.mean = np.asarray([104, 116, 122], dtype=np.float32)

    def __call__(self, x, outputs=['loss3/classifier']):
#        return self.func(inputs={'data': x}, outputs=outputs, disable=['loss1/ave_pool', 'loss2/ave_pool'], train=False)
        return super(GoogleNet, self).__call__(inputs={'data': x}, outputs=outputs, disable=['loss1/ave_pool', 'loss2/ave_pool', 'inception_4d/1x1', 'inception_4d/3x3_reduce', 'inception_4d/5x5_reduce', 'inception_4d/pool'])
