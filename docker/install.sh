#!/usr/bin/bash
cd /home/$USER
# echo "export PATH=/usr/local/cuda/bin:/usr/local/cuda/NsightCompute-2019.1${PATH:+:${PATH}}" >> ~/.zhrc
# echo "export PATH=/usr/local/cuda/bin:/usr/local/cuda/NsightCompute-2019.1${PATH:+:${PATH}}" >> ~/.bashrc
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -p $HOME/miniconda
/bin/bash -c "source /home/$USER/.bashrc"
# source ~/miniconda/etc/profile.d/conda.sh
#conda create -n traindet python=3.7
# eval "$(conda shell.bash hook)"
# source /home/$USER/.bashrc
