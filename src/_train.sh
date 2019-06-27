#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# # Tensorboard
# pkill tensorboard
# # rm -rf logs/tb*
# tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Train tf 
print_header "Training network"
cd $DIR

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

exp='maml-omniglot-5way-1shot-TEST'
dataset='omniglot'
num_cls=5
num_inst=1
batch=1
m_batch=32
num_updates=15000
num_inner_updates=5
lr='5e-1'
meta_lr='1e-2'
gpu=0
python3.6 maml.py $exp --dataset $dataset --num_cls $num_cls --num_inst $num_inst --batch $batch --m_batch $m_batch --num_updates $num_updates --num_inner_updates $num_inner_updates --lr $lr --meta_lr $meta_lr --gpu $gpu 2>&1 | tee ../logs/$exp
