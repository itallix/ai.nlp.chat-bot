# Chat Bot
This is a simple chat bot based on `sentence-transformers/all-MiniLM-L6-v2` model fine-tuned on `kunalbhar/house-md-transcripts` which serves as a BiEncoder and `cross-encoder/ms-marco-MiniLM-L-6-v2` as a CrossEncoder.

![Chat Bot](img/app.png)

## Architecture

![Architecture](img/arch.png)

## Fine-tuning
The model was fine-tuned on the `kunalbhar/house-md-transcripts` dataset using the `sentence-transformers` library. The model was trained for 5 epochs with a batch size of 16 and a learning rate of 2e-5.
WB Report: [link](https://api.wandb.ai/links/vkrnsno/6r91jtbq)

## Loading embeddings into ClickHouse
The embeddings were calculated in `notebooks/train_biencoder.ipynb`, saved as an npy file and pre-loaded into ClickHouse using the `scripts/write_embeddings.py` script.

## Triton Inference Server
Fine-tuned model and reranker are served using Triton Inference Server. Chat Bot uses the `tritonclient` library to interact with the server via gRPC.

## Minio
The model and reranker exported in TorchScript format and stored in Minio.
