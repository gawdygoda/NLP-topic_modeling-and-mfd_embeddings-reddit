import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle
from tqdm import tqdm
import math
import time


# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "roberta-base"
BATCH_SIZE = 1536      # adjust per GPU memory
MAX_LEN = 512         # max tokens per document
OUT_PATH = "embeddings.pkl"
USE_SINGLE_GPU = False # note using multi GPU on Arc is slower by 4x

# Dummy dataset — replace with your list of documents
documents = ["This is the first document.",
             "Here’s another example document.",
             "And a third one for demonstration."] * 33334  # simulate ~3k docs

# ------------------------------
# Start timer
# ------------------------------
start_time = time.time()

# ------------------------------
# Load model & tokenizer
# ------------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)
model.eval()

# Detect GPUs
device_count = torch.cuda.device_count()
if device_count == 0:
    raise RuntimeError("No GPUs detected! Please run on a GPU machine.")

print(f"Using {device_count} GPUs")

# Wrap model with DataParallel
if USE_SINGLE_GPU:
    # Use only the first GPU
    device = torch.device("cuda:0")
else:
    # Use all available GPUs with DataParallel
    if device_count > 1:
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# ------------------------------
# Helper: batch iterator
# ------------------------------
def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# ------------------------------
# Process batches
# ------------------------------
all_embeddings = []
truncated_count = 0

batches = list(batchify(documents, BATCH_SIZE))

for batch in tqdm(batches, desc="Encoding documents"):
    # Tokening twice to count docs that exceed max token length for RoBERTa (512)
    # Tokenize WITHOUT automatic truncation first
    tokenized = tokenizer(batch, return_tensors='pt', padding=True, truncation=False)
    
    # Count truncations
    for doc_tokens in tokenized['input_ids']:
        if len(doc_tokens) > MAX_LEN:
            truncated_count += 1

    # Tokenize
    inputs = tokenizer(batch, return_tensors='pt', padding=True,
                       truncation=True, max_length=MAX_LEN)
    
    # Move inputs to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

    all_embeddings.append(cls_embeddings)

print(f"Number of documents truncated: {truncated_count}")

# Combine all into one tensor
all_embeddings = torch.cat(all_embeddings, dim=0)

# ------------------------------
# Save to Pickle
# ------------------------------
with open(OUT_PATH, "wb") as f:
    pickle.dump(all_embeddings, f)

# ------------------------------
# End timer & report
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

print(f"\n✅ Saved {all_embeddings.shape[0]} embeddings to {OUT_PATH}")
print(f"Each embedding has {all_embeddings.shape[1]} dimensions")
print(f"⏱️ Total execution time: {mins:.0f} min {secs:.2f} sec")