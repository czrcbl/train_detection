FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

### Install Cudagl

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243

ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-10-1 \
        cuda-cudart-10-1 \
cuda-compat-10-1 && \
ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"


# Setup MXNET

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

# Install sudo
RUN apt-get update && \
      apt-get -y install sudo

# Add user
ENV USER=cezar
ENV PASSWD=cezar
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

USER $USER
WORKDIR /home/$USER
# COPY requirements.txt /home/$USER/requirements.txt
# COPY post_install.sh /post_install.sh
# RUN python3 -m virtualenv --python=/usr/bin/python3 /home/$USER/venv
# RUN . /home/$USER/venv/bin/activate && pip install -r /home/$USER/requirements.txt
# RUN mkdir /home/$USER/Projects

RUN echo "export PATH=/usr/local/cuda/bin:/usr/local/cuda/NsightCompute-2019.1${PATH:+:${PATH}}" >> ~/.bashrc
# RUN echo "source /home/$USER/venv/bin/activate" >> ~/.bashrc
RUN /bin/bash -c "source /home/$USER/.bashrc"

# USER $USER
# COPY entrypoint.sh /home/$USER/entrypoint.sh
# RUN chmod +x /home/$USER/entrypoint.sh
# ENTRYPOINT ["/home/$USER/entrypoint.sh"]
