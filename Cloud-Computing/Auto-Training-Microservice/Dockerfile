#FROM python:3.10.0b3-alpine3.14
FROM python:3.7

WORKDIR /app

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements1.txt requirements1.txt

RUN pip3 install -r requirements1.txt

COPY app.py app.py

COPY train.py train.py

COPY templates templates

CMD [ "python3", "app.py"]
