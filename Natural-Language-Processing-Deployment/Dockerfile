FROM frolvlad/alpine-python-machinelearning:latest
COPY . /app
EXPOSE 4000
WORKDIR /app
RUN apk add build-base
RUN apk add --no-cache --virtual .build-deps g++ python3-dev libffi-dev openssl-dev && \
    apk add --no-cache --update python3 && \
    pip3 install --upgrade pip setuptools
RUN pip install - r requirements.txt
CMD app1.py