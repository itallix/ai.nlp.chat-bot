FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  

WORKDIR /app

COPY app.py /app/

RUN pip3 install flask clickhouse-driver numpy transformers tritonclient[all] 

ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/app.py"]