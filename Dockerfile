FROM at-docker:5000/microns-base:cuda11.8.0-python3.8
LABEL maintainer="Stelios Papadopoulos <spapadop@bcm.edu>"

RUN python -m pip install --no-cache-dir --upgrade jupyterlab

RUN pip3 install \
        cloud-volume \
        caveclient \
        nglui \
        slackclient

WORKDIR /
ARG CLOUDVOLUME_TOKEN
RUN mkdir -p .cloudvolume/secrets
RUN echo "{\"token\": \"${CLOUDVOLUME_TOKEN:-}\"}" > .cloudvolume/secrets/cave-secret.json

COPY . /src/microns-coregistration
RUN pip install -e /src/microns-coregistration/python/microns-coregistration
RUN pip install -e /src/microns-coregistration/python/microns-coregistration-api