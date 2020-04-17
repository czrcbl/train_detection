# Object Detection for Additive Manufactured Parts

The rest of this repository contains the code base used on the article(s) [to be published](article_link).

## Setup

### Installing CUDA 10.1
You will need CUDA 10.1 in order to run `mxnet 1.51`, on Ubuntu:

Add Nvidia repository:
```bash
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
```

Install CUDA 10.1
```bash
sudo apt-get update && sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-1 cuda-drivers
```

### Setting up the environment: 

Create a environment (I use conda):
```bash
conda create -n traindet python=3.7.6
conda activate traindet
```
Install packages:
```bash
git clone https://github.com/czrcbl/train_detection
cd train_detection
pip install -r requirements.txt
```

In order to run some scripts, you need to install the following packages:
```bash
git clone https://github.com/czrcbl/bboxes
cd bboxes
pip install -e .

git clone https://github.com/czrcbl/detection
cd detection
pip install -e .
```
### Rendering

In order to run the rendering code, you need the  `blender` executable, version 2.80, in the path.
Download from [HERE](https://www.blender.org/download).
## Structure

* Folder `traindet` has the core utilities from training and evaluating
the models.
* Folder `rendering` has all the code that is supposed to be run through blender. You must have the blender executable on system path.
* Folder `scripts` has the scripts for launching the networks training, starting the rendering of synthetic images and producing some visualizations.
* All the data necessary to train the models should be placed on the `data` folder.

## Training a model

Just call the model train scrip with the required arguments, models are saved under `data/chackpoints/<dataset_name>/<model_name>`:
Some example calls are on the `command.py` file.
