#!/usr/bin/bash
$containerName=cuda101man
docker run -it \
    --mount type=bind,source=/home/$USER,target=/home/$USER \
    --gpus all \
    --name cuda101man \
    czrcbl/dev:cuda101man
