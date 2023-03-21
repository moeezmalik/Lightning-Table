FROM python:3.7

WORKDIR /usr/lightning

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

ENV TERM "xterm"