#! /usr/bin/bash
cd /home/$USER
git clone https://github.com/czrcbl/bboxes
cd bboxes
pip install -e .
cd ..
git clone https://github.com/czrcbl/detection
cd detection
pip install -e .
cd ..