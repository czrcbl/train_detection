# Object Detection for Additive Manufactured Parts

## Intro

This repository contains the code necessry to reproduce the results obtained on
the article [to be published](article_link).

## Structure


* Folder `traindet` has the core functionalities from training and evaluating
the models.
* Folder `rendering` has all the code that is supposed to be run through blender
* Folder `scripts` has the scripts for launching the networks training and also
from start the rendering of systetic images


## Reproducing

Execute:

``` bash
python scripts/train_experiments.py --test --model all
```

To train all models for 1 epoch. Remove the --test flag to perform the complete
training

Then, run:

``` bash
python traindet/main.py
```

To apply the models to the validation set, perform the benchmarks and store the results on the `gen_data`
folder.

In the folder `nbs` there notebooks that compile the results.
The notebook `results.ipynb` contains the steps to compile the tables and metrics.
The notebook `visualization` contains a visualization of the validation.
