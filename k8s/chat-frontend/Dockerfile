FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  

WORKDIR /app

COPY app.py /app/
COPY house.jpg /

RUN pip3 install gradio requests

ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/app.py"]
