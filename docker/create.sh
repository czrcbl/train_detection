#!/usr/bin/bash

docker run -it \
    --mount type=bind,source=/home/$USER/Docker/traindet,target=/home/user \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name traindet \
    czrcbl/dev:traindet102

#export containerId=$(docker ps -l -q)
#xhost +local:'docker inspect --format='' $containerId'
#docker start -a -i $containerId
