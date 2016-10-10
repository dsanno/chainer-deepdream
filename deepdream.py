import argparse
import os
import numpy as np
from scipy import ndimage as nd
from PIL import Image
import chainer
from chainer import Variable, cuda, functions as F
from net import GoogleNet

parser = argparse.ArgumentParser(description='DCGAN trainer for ETL9')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='bvlc_googlenet.caffemodel', type=str,
                    help='googlenet caffe model file path')
parser.add_argument('--image', '-i', required=True, type=str,
                    help='image file path')
parser.add_argument('--output_dir', '-o', default='.', type=str,
                    help='directory to output images')
parser.add_argument('--size', '-s', default=-1, type=int,
                    help='output image size')
parser.add_argument('--iter', default=100, type=int,
                    help='number of iteration')
args = parser.parse_args()
xp = np

def objective(dest):
    return 0.5 * F.sum(dest ** 2)

def update_step(net, images, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    offset_x, offset_y = np.random.randint(-jitter, jitter + 1, 2)
    data = np.roll(np.roll(images, offset_x, -1), offset_y, -2)

    x = Variable(xp.asarray(data))
    x.zerograd()
    dest, = net(x, outputs=[end])
    objective(dest).backward()
    g = cuda.to_cpu(x.grad)

    data[:] += step_size / np.abs(g).mean() * g
    data = np.roll(np.roll(data, -offset_x, -1), -offset_y, -2)
    if clip:
        bias = net.mean.reshape((1, 3, 1, 1))
        data[:] = np.clip(data, -bias, 255 - bias)
    return data

def update(net, base_image, step_num=10, octave_num=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    x = net.preprocess(base_image)
    octaves = [x.reshape((1,) + x.shape)]

    for i in range(octave_num - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)
        x = octave_base + detail
        for i in xrange(step_num):
            x = update_step(net, x, end=end, clip=clip, **step_params)
        detail = x[0] - octave_base
    return net.deprocess(x[0])

image = Image.open(args.image).convert('RGB')
if args.size > 0:
    w, h = image.size
    scale = float(args.size) / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    image = image.resize(new_size, Image.BILINEAR)
image = np.float32(image)
output_dir = args.output_dir
try:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir):
        print 'Error: output path is not directory: {}'.format(output_dir)
        exit()
except:
    print 'Error: cannot make dir: {}'.format(output_dir)
    exit()
net = GoogleNet(args.model)
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    xp = cuda.cupy
    net.to_gpu(args.gpu)

scale = 0.05
h, w = image.shape[:2]
for iteration in range(args.iter):
    image = update(net, image)
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(output_dir, 'deepdream_{0:03d}.png'.format(iteration)))
    image = nd.affine_transform(image, [1 - scale, 1 - scale, 1], [h * scale / 2, w * scale / 2, 0], order=1)
    print 'iteration {} done'.format(iteration + 1)
