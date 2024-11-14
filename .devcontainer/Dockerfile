FROM ubuntu:18.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y sudo wget git-core rsync curl net-tools libxml2 libgl1-mesa-glx \
    && apt-get clean

ARG USERNAME="vscode"
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG HOME=/home/$USERNAME
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


SHELL [ "/bin/bash", "-c" ]

# Install required libraries and python
RUN apt-get update && \
    apt-get install -y libnfft3-dev vim build-essential software-properties-common libpython3.6-dev python3 wget python3-pip cmake qt5-default


RUN pip3 install --upgrade pip && pip3 install progressbar joblib pyqt5 pyqtgraph matplotlib && \ 
    pip3 install git+https://github.com/chaithyagr/twixreader

# Install CUDA 10.1 Update 2
RUN wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run && \
sh cuda_10.1.243_418.87.00_linux.run --toolkit --silent

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NO_FMM_BUILD=1
ENV PYTHONPATH="/volatile/Chaithya/Environments/RenataEnv/lib64/python3.6/site-packages"






