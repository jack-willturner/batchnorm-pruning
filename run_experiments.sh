#!/bin/bash
# qsub options:
#$ -l gpu=1
#$ -q gpgpu
# First option informs scheduler that the job requires a gpu.
# Second ensures the job is put in the batch job queue, not the interactive queue
 
# Set up the CUDA environment
export CUDA_HOME=/opt/cuda-8.0.44
export CUDNN_HOME=/opt/cuDNN-5.1
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
 
export PYTHON_PATH=$PATH
# Activate the relevant virtual environment:
source activate exp
 
#####
# MAKE SURE TO USE THIS, WHATEVER ENVIRONMENT YOU ARE USING.
# The path should point to the location where you saved the gpu_lock_script provided below.
# It's important to include this, as it prevents collisions on the GPUs.
source ~/gpu_lock_script.sh

python main.py
#End of template
