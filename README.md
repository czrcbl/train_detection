# Object Detection for Additive Manufactured Parts

## Intro

This repository contains the code necessry to reproduce the results obtained on
the article [to be published](article_link).

## Structure

* Folder `traindet` has the core utlities from training and evaluating
the models.
* Folder `rendering` has all the code that is supposed to be run through blender.
* Folder `scripts` has the scripts for launching the networks training and also
for starting the rendering of synthetic images.
* All the data to train the models should be placed on the  `data` folder.
All generated data is also placed in this folder.

## Reproducing

Execute:

``` bash
python scripts/train_experiments.py --test --model all
```

To train all models for 1 epoch. Remove the --test flag to perform the complete
training

Then, run:

``` bash
python scripts/make_predictions.py --modeel all --dataset all
```

To apply the models to the validation set and store the results on the `data/gen_data`
folder.

In the folder `nbs` there will be notebooks that compile the results.
