import torch
# from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm
import math
import time
import pandas as pd


# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "roberta-base"
BATCH_SIZE = 1536      # adjust per GPU memory
MAX_LEN = 512         # max tokens per document
STRIDE = 128
OUT_PATH = "embeddings.pkl"
USE_SINGLE_GPU = True # note using multi GPU on Arc is slower by 4x
DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/aita_filtered.pkl"
MODEL_NAME = "roberta-base"
# MODEL_NAME = "microsoft/mpnet-base"

# # Dummy dataset — replace with your list of documents
# documents = ["This is the first document.",
#              "Here’s another example document.",
#              "And a third one for demonstration."] * 33334  # simulate ~3k docs

raw_data = pd.read_pickle(DATA_FILE)
print(f'Read in {raw_data.shape[0]} posts')
# documents = raw_data["selftext"].tolist()
documents = raw_data["selftext"].iloc[:1000].tolist()

# ------------------------------
# Start timer
# ------------------------------
start_time = time.time()

# ------------------------------
# Load model & tokenizer
# ------------------------------
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
# model = RobertaModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
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
total_chunks = 0
truncated_count = 0

batches = list(batchify(documents, BATCH_SIZE))

for batch in tqdm(batches, desc="Encoding documents"):
    # # Tokening twice to count docs that exceed max token length for RoBERTa (512)
    # # Tokenize WITHOUT automatic truncation first
    # tokenized = tokenizer(batch, return_tensors='pt', padding=True, truncation=False)
    
    # # Count truncations
    # for doc_tokens in tokenized['input_ids']:
    #     if len(doc_tokens) > MAX_LEN:
    #         truncated_count += 1

    # Tokenize
    inputs = tokenizer(batch, 
                        return_tensors='pt', 
                        padding=True,
                        truncation=True,
                        max_length=MAX_LEN,
                        stride=STRIDE,
                        return_overflowing_tokens=True
    )

    overflow_mapping = inputs.pop("overflow_to_sample_mapping")
    attention_mask = inputs['attention_mask']

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.cpu()
    
    # ---------------------------------------------------------
    # Reconstruct Documents (Handling Overlap)
    # ---------------------------------------------------------
    
    # Temporary storage for the current batch
    # We use a dictionary because chunks might be out of order in rare cases
    # structure: { doc_index: [ (chunk_idx, tensor), ... ] }
    doc_chunks = {}
    
    # Identify which chunk number this is for the specific doc
    # e.g., if doc 0 has 3 chunks, we need to know which is #1, #2, #3
    current_doc_id = -1
    chunk_counter = 0

    for i, doc_id_tensor in enumerate(overflow_mapping):
        doc_id = doc_id_tensor.item()
        
        # Determine chunk index (0, 1, 2...)
        if doc_id != current_doc_id:
            current_doc_id = doc_id
            chunk_counter = 0
        else:
            chunk_counter += 1
            
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
            
        doc_chunks[doc_id].append((chunk_counter, hidden_states[i], attention_mask[i]))

    # Now stitch them together
    # We iterate through the documents found in this batch
    sorted_doc_ids = sorted(doc_chunks.keys())
    
    for doc_id in sorted_doc_ids:
        chunks = doc_chunks[doc_id]
        # Sort by chunk_counter to ensure order (0, 1, 2)
        chunks.sort(key=lambda x: x[0])
        
        stitched_tokens = []
        
        for chunk_idx, embedding, mask in chunks:
            # embedding shape: [512, 768]
            # mask shape: [512]
            
            # 1. Identify valid tokens (remove Padding)
            # mask is 1 for tokens, 0 for padding
            valid_len = mask.sum().item()
            
            # Slice to remove padding
            # But wait! Special tokens (<s> and </s>) are "valid" in the mask
            # valid_len usually includes <s> at start and </s> at end
            
            # 2. Slice based on chunk position
            if chunk_idx == 0:
                # First Chunk: Keep Start, Remove End special token
                # We usually want to remove <s> (index 0) and </s> (index valid_len-1)
                # Let's keep words only: [1 : valid_len-1]
                relevant_tokens = embedding[1 : valid_len-1]
                
            else:
                # Subsequent Chunks: 
                # Remove Start special token (1)
                # Remove Overlap (STRIDE)
                # Remove End special token (1)
                
                start_cut = 1 + STRIDE
                end_cut = valid_len - 1
                
                if start_cut < end_cut:
                    relevant_tokens = embedding[start_cut : end_cut]
                else:
                    relevant_tokens = torch.tensor([]) # Handle edge case where stride covers whole remaining text
            
            stitched_tokens.append(relevant_tokens)
            
        # Stack all tokens for this document
        if stitched_tokens:
            full_doc_tensor = torch.cat(stitched_tokens, dim=0)
            all_embeddings.append(full_doc_tensor)
        else:
            all_embeddings.append(torch.zeros(0, 768))

# print(f"Number of documents truncated: {truncated_count}")

# # Combine all into one tensor
# all_embeddings = torch.cat(all_embeddings, dim=0)

# ------------------------------
# Save to Pickle
# ------------------------------
with open(OUT_PATH, "wb") as f:
    pickle.dump(all_embeddings, f)

print(f"\nSaved {len(all_embeddings)} embeddings to {OUT_PATH}")
# Print the shape of the first 10 posts
for i in range(10):
    # Check if index exists (in case you have <10 docs)
    if i < len(all_embeddings):
        print(f"Post {i}: {all_embeddings[i].shape}")
# ------------------------------
# End timer & report
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"Total execution time: {mins:.0f} min {secs:.2f} sec")
