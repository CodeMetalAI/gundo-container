FROM arm64v8/python:3.8 as build

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETOS
ARG TARGETARCH

ARG Version
ARG GitCommit
RUN echo "I am running on $BUILDPLATFORM, building for $TARGETPLATFORM" 


COPY requirements.txt requirements.txt
RUN python3.8 -m pip install --upgrade pip
RUN cat requirements.txt | xargs -n 1 -L 1 python3 -m pip install
COPY . .

CMD ["python3.8", "main.py"]
