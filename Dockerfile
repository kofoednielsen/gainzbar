FROM nvidia/cuda:10.2-base

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

run apt update
run apt install python3 -y
run apt install python3-pip -y

COPY ./requirements.txt /app/
WORKDIR app
run python3 -m pip install -r requirements.txt
run apt install -y libsm6 libxext6 libxrender-dev
COPY ./gainzbar /app
COPY ./faces/ /app/faces/

CMD python3 ai.py
