FROM ubuntu:16.04

MAINTAINER Jeremy Magland

# Python3
RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip

# Python3 packages
RUN pip3 install --upgrade numpy
RUN pip3 install --upgrade numpydoc

ADD . /package

# Build
WORKDIR /package
RUN processors/processors.mp spec > processors.spec
