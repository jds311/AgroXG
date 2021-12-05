#FROM python:3.10.0b3-alpine3.14
FROM python:3.7

WORKDIR /app

COPY requirement.txt requirement.txt

RUN pip3 install -r requirement.txt

COPY app.py app.py

COPY templates templates

COPY static static

CMD [ "python3", "app.py"]
