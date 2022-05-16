FROM at-docker.ad.bcm.edu:5000/microns-base 
LABEL maintainer="Stelios Papadopoulos <spapadop@bcm.edu>"

RUN pip3 install \
        cloud-volume \
        caveclient \
        nglui \
        slackclient

WORKDIR /root
ARG CLOUDVOLUME_TOKEN
RUN mkdir -p .cloudvolume/secrets
RUN echo "{\"token\": \"${CLOUDVOLUME_TOKEN:-}\"}" > .cloudvolume/secrets/cave-secret.json

# ALLEN INSTITUTE
RUN pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3

# MISC PACKAGES
RUN pip3 install git+https://github.com/spapa013/wridgets.git

# CURRENT PACKAGE
# TODO: torch rebuilds partially after edit, consider improving docker caching (e.g. maybe with requirements.txt install in separate step)
COPY . /src/microns-coregistration
RUN pip install --prefix=$(python -m site --user-base) -e /src/microns-coregistration/python/microns-coregistration
RUN pip install --prefix=$(python -m site --user-base) -e /src/microns-coregistration/python/microns-coregistration-api