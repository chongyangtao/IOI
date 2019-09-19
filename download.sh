#!/usr/bin/env bash

PWD=$(pwd)

UBUNTU_DIR=$PWD/data/ubuntu
mkdir -p $UBUNTU_DIR

# Download Ubuntu dialogue corpus
wget https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=1 -O $UBUNTU_DIR/ubuntu_data.zip
unzip -j $UBUNTU_DIR/ubuntu_data.zip -d $UBUNTU_DIR

