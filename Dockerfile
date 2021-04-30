FROM python:3.8.5

WORKDIR /usr/src/pipeline

COPY . .

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]