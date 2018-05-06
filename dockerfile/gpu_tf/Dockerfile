FROM nvidia/cuda:9.0-cudnn7-devel

COPY requirements.txt /home
RUN apt-get -y update
RUN apt-get install -y python3 python3-pip nano

# Dependent packages for pillow
RUN apt-get install -y zlib1g-dev libjpeg-dev libncurses5-dev

# [6 May] Downgrade cuDNN to 7.0.5 to ensure compatibility with tensorflow and cuda 9.0
RUN apt-get install -y --allow-downgrades libcudnn7=7.0.5.15-1+cuda9.0 libcudnn7-dev=7.0.5.15-1+cuda9.0 

# Use pip to install other cs231n dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r /home/requirements.txt
RUN pip3 install --upgrade jupyter
RUN pip3 install --upgrade tensorflow-gpu
