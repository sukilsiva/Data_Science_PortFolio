FROM frolvlad/alpine-python-machinelearning:latest
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN apk add build-base
RUN apk add --no-cache --virtual .build-deps g++ python3-dev libffi-dev openssl-dev && \
    apk add --no-cache --update python3 && \
    pip3 install --upgrade pip setuptools
RUN pip install - r requirements.txt
CMD app.py
