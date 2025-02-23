import os
import numpy as np
import pandas as pd

from clickhouse_driver import Client

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 9000
CLICKHOUSE_USER = "clickhouseuser"
CLICKHOUSE_PASSWORD = "clickhousepassword"

if __name__ == "__main__":
    client = Client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD
    )
    client.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id UInt64,
        sentence String,
        embedding Array(Float32)
    ) ENGINE = MergeTree()
    ORDER BY id
    ''')
    embeddings = np.load("embeddings.npy")
    sentences = pd.read_csv("sentences.csv", skiprows=1, header=None)[0].tolist()
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        if i % 1000 == 0:
            print(f"Inserting {i}th row\n")
        client.execute(
            "INSERT INTO embeddings (id, sentence, embedding) VALUES",
            [(i, sentence, embedding.tolist())]
        )
    client.disconnect()
