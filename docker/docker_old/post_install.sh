#!/usr/bin/bash
cd /home/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -p $HOME/miniconda
source ~/.bashrc
# source ~/miniconda/etc/profile.d/conda.sh
#conda create -n traindet python=3.7
# eval "$(conda shell.bash hook)"
# source /home/$USER/.bashrc
