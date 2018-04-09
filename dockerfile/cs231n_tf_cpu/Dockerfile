# Base image Ubuntu:16.04
FROM ubuntu:16.04

COPY requirements.txt /home
RUN apt-get -y update
RUN apt-get install -y python3 python3-pip nano
# Dependent packages for pillow
RUN apt-get install -y zlib1g-dev libjpeg-dev libncurses5-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r /home/requirements.txt
RUN pip3 install --upgrade jupyter
RUN pip3 install tensorflow
