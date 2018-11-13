#!/bin/sh
export PATH="$HOME/miniconda/bin:$PATH"

source activate bertie

cd ..
python train_resnet.py --GPU=0,1 --workers=0

source deactivate 
