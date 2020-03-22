FROM nvidia/cuda:10.2-base

run apt update
run apt install python3 -y
run apt install python3-pip -y

COPY ./requirements.txt /app/
WORKDIR app
run python3 -m pip install -r requirements.txt
COPY ./gainzbar /app
COPY ./faces/ /app/faces/

CMD python3 ai.py
