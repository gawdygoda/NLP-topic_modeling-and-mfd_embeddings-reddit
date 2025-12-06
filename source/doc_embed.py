import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------

# MODEL_NAME = "microsoft/mpnet-base"
MODEL_NAME = "roberta-base"

MAX_LEN = 512

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "aita_filtered.pkl")
# OUT_EMB_FILE = os.path.join(DATA_DIR, "doc_embeddings_mpnet_base.npy")
OUT_EMB_FILE = os.path.join(DATA_DIR, "doc_embeddings_roberta_base.npy")


# ------------------------------
# Definitions
# ------------------------------
def embed_doc(text: str,
                   tokenizer: AutoTokenizer,
                   model: AutoModel,
                   device: torch.device,
                   max_len: int = 512) -> torch.Tensor:
    """
    Returns a single embedding for a document.

    Chunk if doc exceeds MAX_LEN

    Uses the CLS token embedding for each chunk.
    """

    # Get token IDs without special tokens- Google Overview
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Handle empty docs- Google Overview
    if len(token_ids) == 0:
        hidden_size = model.config.hidden_size
        return torch.zeros(hidden_size)

    chunk_embs = []

    for start in range(0, len(token_ids), max_len):
        chunk_ids = token_ids[start:start + max_len]

        # Turn chunk of ids back into text
        chunk_text = tokenizer.decode(chunk_ids)

        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

        # Move to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # From Google Overview
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token = first token's hidden state
            # shape: [1, hidden_dim]
            cls_emb = outputs.last_hidden_state[:, 0, :]

        # Squeeze to [hidden_dim]
        chunk_embs.append(cls_emb.squeeze(0).cpu())

    # Average chunks
    if len(chunk_embs) == 1:
        return chunk_embs[0]
    else:
        return torch.stack(chunk_embs, dim=0).mean(dim=0)


# ------------------------------
# Main
# ------------------------------
def main():
    start_time = time.time()

    # Load AITA file
    print(f"Loading data from {DATA_FILE} ...")
    df = pd.read_pickle(DATA_FILE)
    print(f"Read {df.shape[0]} posts")

    docs = df["selftext"].fillna("").tolist()

    # Load model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    embeddings = []
    num_docs = len(docs)

    print(f"Encoding {num_docs} documents with chunking (max_len={MAX_LEN})...")
    for text in tqdm(docs, desc="Encoding docs"):
        doc_emb = embed_doc(text, tokenizer, model, device, MAX_LEN)
        embeddings.append(doc_emb)

    # From ChatGPT
    print("Stacking embeddings...")
    embeddings = torch.stack(embeddings, dim=0)   # [N_docs, hidden_dim]
    emb_np = embeddings.numpy().astype("float16")

    print(f"Final embedding matrix shape: {emb_np.shape}")
    print(f"Saving embeddings to {OUT_EMB_FILE} ...")
    np.save(OUT_EMB_FILE, emb_np)

    # End timer & report
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Total execution time: {mins:.0f} min {secs:.2f} sec")


if __name__ == "__main__":
    main()
