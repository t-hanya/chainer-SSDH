#!/bin/bash

if [ ! -d data ]; then
    mkdir data
fi
cd data

# download pre-trained parameter file for AlexNet
echo "Download the pre-trained AlexNet model"
wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

# download mean image file
echo "Download the mean image file"
wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy --no-check-certificate
