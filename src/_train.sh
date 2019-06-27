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

python3.6 maml.py \
--env-name "Regression-v0" \
--seed 0 \
--learner-type "meta" \
--policy-type "continuous" \
--ep-max-timesteps 100 \
--n-traj 1 \
--meta-batch-size 32 \
--fast-num-update 5 \
--meta-lr 0.01 \
--fast-lr 0.5 \
--n-agent 1 \
--prefix ""
