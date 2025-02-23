import logging
import os
import numpy as np
import tritonclient.grpc as grpcclient

from clickhouse_driver import Client
from flask import Flask, request
from transformers import AutoTokenizer

BIENCORDER_TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_TOKENIZER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_BIENCODER_ID = "biencoder"
LOCAL_RERANKER_ID = "reranker"
EMBEDDING_DIM = 384

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
biencoder_tokenizer = AutoTokenizer.from_pretrained(BIENCORDER_TOKENIZER_ID)
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_TOKENIZER_ID)
inference_client = grpcclient.InferenceServerClient(
    url=f"{os.getenv('INFERENCE_HOST', 'localhost')}:{os.getenv('INFERENCE_PORT', '8001')}"
)
clickhouse_client = Client(
    host=os.getenv("CLICKHOUSE_HOST", "localhost"), 
    port=os.getenv("CLICKHOUSE_PORT", 9000),
    user=os.getenv("CLICKHOUSE_USER", "clickhouseuser"), 
    password=os.getenv("CLICKHOUSE_PASSWORD", "clickhousepassword")
)

def get_embedding(
    client: grpcclient.InferenceServerClient, 
    text: str    
) -> np.ndarray:
    inputs = biencoder_tokenizer(text, padding=True, truncation=True, max_length=EMBEDDING_DIM, return_tensors="np")
    return _infer(client, inputs, LOCAL_BIENCODER_ID)

def get_ranking(
    client: grpcclient.InferenceServerClient,
    query: str,
    text: str    
) -> np.ndarray:    
    inputs = reranker_tokenizer(query, text, truncation=True, padding="max_length", max_length=EMBEDDING_DIM, return_tensors="np")
    return _infer(client, inputs, LOCAL_RERANKER_ID)
    
def _infer(
    client: grpcclient.InferenceServerClient, 
    inputs: dict,
    model_name: str
) -> np.ndarray:
    input_ids = grpcclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64")
    attention_mask = grpcclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")

    input_ids.set_data_from_numpy(inputs["input_ids"])
    attention_mask.set_data_from_numpy(inputs["attention_mask"])

    output = grpcclient.InferRequestedOutput("output")

    response = client.infer(
        model_name=model_name,
        inputs=[input_ids, attention_mask],
        outputs=[output]
    )

    return response.as_numpy("output")

@app.route('/', methods=['POST'])
def get_closest():
    data = request.get_json()
    text = data.get('text', '')
    embedding = get_embedding(inference_client, text)
    ch_emb = embedding.squeeze().tolist()
    result = clickhouse_client.execute("""
        SELECT
	        sentence,
	        cosineDistance(%(my_embedding)s, embedding) AS score
        FROM embeddings
        ORDER BY score ASC
        LIMIT 5
    """, {"my_embedding": ch_emb})
    
    scores = []
    for row in result:
        logger.info("sentence: %s", row)
        score = get_ranking(inference_client, text, row[0])
        logger.info("rank: %s", score)
        scores.append((row[0], score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True) 
    logger.info("sorted ranks: %s", scores)
    return scores[0][0]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
