FROM python:3.7

WORKDIR /usr/lightning

COPY . .

RUN pip install -r requirements.txt

CMD ["make", "evaluation-complete"]