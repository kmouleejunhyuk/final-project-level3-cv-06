FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04 
MAINTAINER keyog <seojiyu1113@gmail.com> 
RUN echo 'export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}' >> ~/.bashrc 
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc 
RUN echo 'export PATH=/usr/local/cuda/bin:/$PATH' >> ~/.bashrc 
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc 
RUN apt-get update 
RUN apt-get install -y vim 
RUN apt-get install -y git 
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH
EXPOSE 8003
EXPOSE 8004
RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install libgl1-mesa-glx -y
