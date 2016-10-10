# Deep Dreams implementation using Chainer

## Requirement

* Python
* [Chainer](http://chainer.org/)
* [Pillow](https://pillow.readthedocs.org/en/3.0.x/)

## Usage

### Download googlenet caffe model

Download bvlc_googlenet.caffemodel from https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet, and put it into repository root.

### Run script

```
$ python deepdream.py -g 0 -i sample.png -o out_image
```

* -g n: GPU device index, or -1 to run without GPU.
* -i file_path: Input image file path.
* -o dir_path: Output image directory path.
* -m file_path: Caffe model file path. Default is "bvlc_googlenet.caffemodel".
* -s size: Output image size. Image will be zoomed so that the larger side length equals this value.
* --iter n: Number of iterations. Default is 100.

## Generated image sample

Original image  
<img src="https://raw.githubusercontent.com/dsanno/chainer-deepdream/master/image/original.png" width="484px alt="generated image sample">  
Generated after 20 iterations  
<img src="https://raw.githubusercontent.com/dsanno/chainer-deepdream/master/image/generated.png" width="484px alt="generated image sample">  
