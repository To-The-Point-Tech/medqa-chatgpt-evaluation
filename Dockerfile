FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential \
                       ca-certificates \
                       wget \
                       curl \
                       unzip \
                       ssh \
                       git \
                       vim \
                       jq

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TOKENIZERS_PARALLELISM="true"

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9-dev python3-pip python3-setuptools
RUN apt-get clean
RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN python -m pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /medqa
COPY . /medqa

RUN pip cache purge