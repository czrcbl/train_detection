# Object Detection for Additive Manufactured Parts

## Detection

`detection` package consists in a wrapper around the trained models and provides some utilities related to the manipulation and visualization of bounding boxes. Detection is self contained and does not depend on the other packages in this repository. The only dependence are the models stored in `data/checkpoints` folder.
This package is supposed to work with python2 and python3. 

<!--
Download models from the [Dropbox Folder](https://www.dropbox.com/sh/1v53pmryf6jrig9/AAAhRwnVNLHnF3vs_eJttB2sa?dl=0). Place them on models folder.
-->


## Other packages

The rest of this repository contains the code necessary to reproduce the results obtained on
the article [to be published](article_link).

## Structure

* Folder `traindet` has the core utilities from training and evaluating
the models.
* Folder `rendering` has all the code that is supposed to be run through blender.
* Folder `scripts` has the scripts for launching the networks training, starting the rendering of synthetic images and producing some visualizations.
* All the data to train the models should be placed on the `data` folder.
All generated data and trained models are also placed in this folder.

## Reproducing

Execute:

``` bash
python scripts/train_experiments.py --test --model all
```

To train all models for 1 epoch. Remove the --test flag to perform the complete
training

Then, run:

``` bash
python scripts/make_predictions.py --model all --dataset all
```

To apply the models to the validation set and store the results on the `data/gen_data`
folder.

In the folder `nbs` there will be notebooks that compile the results.
