FROM unlhcc/cuda-ubuntu:10.2
LABEL Rahul Prajapati <rahul.prajapati90904@gmail.com>

#github https://github.com/LordVoldemort28/docker-deep-machine-learning

#Credit - waleedka/modern-deep-learning
#https://hub.docker.com/r/waleedka/modern-deep-learning/ 

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    apt-utils \ 
    git \ 
    curl \ 
    zsh \
    vim \ 
    unzip \ 
    wget \
    build-essential cmake \ 
    libopenblas-dev 

# Python 3.7
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y python3.7 python3.7-dev \
    && cd /usr/bin \
    && ln -s /usr/bin/python3.7 python 

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && echo "alias python3='python3.7'" >> /root/.bash_aliases \
    && echo "alias pip='pip3'" >> /root/.bash_aliases \
    && /bin/bash -c "source /root/.bash_aliases" \
    && pip3 install --upgrade pip

# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install 'pillow<7'

# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image pandas matplotlib requests

# Install PyTorch (and friends) for both Python 3.5
RUN pip3 --no-cache-dir install 'torchvision==0.7.0' 'torch==1.6.0' torchsummary scikit-learn 'networkx==2.0'

# Java
RUN apt-get install -y --no-install-recommends default-jdk

# Cython
RUN pip3 install --no-cache-dir --upgrade Cython cython

#Streamlit & Pretty table
RUN pip3 install streamlit PrettyTable

#Tensorboard
RUN pip3 install --no-cache-dir --upgrade tensorboard 

#Comet-ml
RUN pip3 install --no-cache-dir --upgrade comet-ml

# Expose port for TensorBoard
EXPOSE 6006

WORKDIR "/root"
CMD ["/bin/bash"]
