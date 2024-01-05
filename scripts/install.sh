#!/usr/bin/env bash

echo "THIS SCRIPT IS WIP AND HAS NOT BEEN TESTED!"

REPO_DIR=$(cd .. && pwd)
PYTORCHBIN_DIR="$REPO_DIR/pytorch_bin"

install_pytorch() {
    cd $REPO_DIR
    echo $(pwd)
    mkdir $PYTORCHBIN_DIR

    # Install dependencies
    echo "Installing dependencies"
    apt update
    apt install python3 python3-pip libopenblas-base libopenmpi-dev libomp-dev
    pip3 install Cython numpy

    echo "Downloading and installing PyTorch"
    cd $PYTORCHBIN_DIR
    wget https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl
    pip3 install torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl

    # Install torchvision
    echo "Downloading and building torchvision. This might take some time."
    apt install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
    git clone --branch release/0.16 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.16.2
    python3 setup.py install --user
}

install_pytorch
