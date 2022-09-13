FROM python:3.10.5

LABEL \
  maintainer="Adil Rashitov <adil@wastelabs.co>" \
  org.opencontainers.image.title="research_kedro_microservice" \
  org.opencontainers.image.description="Docker image for research_kedro_microservice project" \
  org.opencontainers.image.authors="Adil Rashitov <adil@wastelabs.co>" \
  org.opencontainers.image.url="https://github.com/AtmosOne/research_kedro_microservice"

WORKDIR /application


COPY ./ ./

ARG aws_access_key_id
ARG aws_secret_access_key
ARG google_maps_key
ENV aws_access_key_id=$aws_access_key_id
ENV aws_secret_access_key=$aws_secret_access_key
ENV google_maps_key=$google_maps_key


RUN /bin/bash -c "pip3 install -r src/requirements.txt"
RUN /bin/bash -c "pip3 uninstall boto3 aiobotocore -y"
RUN /bin/bash -c "pip3 install boto3==1.24.13"
RUN /bin/bash -c "pip3 install aiobotocore==2.3.3"

CMD /bin/bash -c "python3 conf/context_management/main.py"

EXPOSE 8000
