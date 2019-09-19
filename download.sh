#!/usr/bin/env bash

PWD=$(pwd)

UBUNTU_DIR=$PWD/data/ubuntu1
mkdir -p $UBUNTU_DIR

# Download Ubuntu dialogue corpus
wget https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=1 -O $UBUNTU_DIR/ubuntu_data.zip
unzip -j $UBUNTU_DIR/ubuntu_data.zip -d $UBUNTU_DIR

# Download pre-trained word embedding file
wget https://www.dropbox.com/s/0ihosy699ga3nie/ubuntu.200d.word2vec.zip?dl=1 -O $UBUNTU_DIR/ubuntu.200d.word2vec.zip
unzip -j $UBUNTU_DIR/ubuntu.200d.word2vec.zip -d $UBUNTU_DIR
