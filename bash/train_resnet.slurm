#!/bin/sh
#SBATCH --gres=gpu:2

export PATH="$HOME/miniconda/bin:$PATH"

nvidia-smi
echo 'Waking up Bertie...'
source activate bertie

cd ..

echo 'Bertie is going to help us train a ResNet on CIFAR-10...'
python train_resnet.py --GPU=0,1 --print_freq=51 --workers=0 --no_epochs=40 --batch_size=256

source deactivate 
