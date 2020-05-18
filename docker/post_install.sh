#!/usr/bin/bash
cd /home/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -p $HOME/miniconda
source ~/.bashrc
# source ~/miniconda/etc/profile.d/conda.sh
conda create -n traindet python=3.7
conda activate traindet
# source /home/$USER/.bashrc
cd /home/$USER/Projects
git clone https://github.com/czrcbl/bboxes
cd bboxes
pip install -e .
cd ..
git clone https://github.com/czrcbl/detection
cd detection
pip install -e .
cd ..