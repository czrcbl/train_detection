FROM nvidia/cudagl:10.1-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install sudo
RUN apt-get update && \
      apt-get -y install sudo

# Add user
ENV USER=user
ENV PASSWD=user
ENV UID=1000
ENV GID=1000

RUN groupadd $USER && \
    useradd --create-home --no-log-init -g $USER $USER && \
    usermod -aG sudo $USER && \
    echo "$PASSWD:$PASSWD" | chpasswd && \
    chsh -s /bin/bash $USER && \
    # Replace 1000 with your user/group id
    usermod  --uid $UID $USER && \
    groupmod --gid $GID $USER

RUN apt install -y git \
    wget \
    nano \
    python3-pip \
    python-virtualenv 

COPY post_install.sh /post_install.sh
USER $USER
WORKDIR /home/$USER
