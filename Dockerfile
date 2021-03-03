FROM python:3.7.5-slim

#RUN apt-get update -y && \
#	apt-get install -y libgl1-mesa-glx && \
#	apt-get install libgtk2.0-dev

RUN apt-get update -y && \
	apt-get install -y libgtk2.0-dev && \
	apt-get install -y ffmpeg libsm6 libxext6

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip install -r requirements.txt

COPY . /

ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]