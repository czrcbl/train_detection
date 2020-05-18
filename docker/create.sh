#!/usr/bin/bash
docker run -it \
    --mount type=bind,source=/home/$USER/traindet_docker,target=/home/user \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name traindet \
    traindet

export containerId=$(docker ps -l -q)
xhost +local:'docker inspect --format='' $containerId'
docker start -a -i $containerId
