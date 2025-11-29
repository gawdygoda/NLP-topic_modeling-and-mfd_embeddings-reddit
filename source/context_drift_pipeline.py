import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
# import pickle
from tqdm import tqdm
# import math
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Only needed temp
import random

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "../data"
RESULTS_DIR = "../results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_SINGLE_GPU = True # note using multi GPU on Arc is slower by 4x
DATA_FILE = f"{DATA_DIR}/aita_filtered.pkl"
MFD_FILE = f"{DATA_DIR}/mfd2.0.dic"

#MODEL_NAME = "roberta-base" # 15min on H200
MODEL_NAME = "microsoft/mpnet-base" # 20min on H200
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME_SAFE = MODEL_NAME.replace("/", "-")
HEATMAP_OUTFILE_YA = f'{RESULTS_DIR}/heatmap_mfd_YA_{MODEL_NAME_SAFE}.png'
HEATMAP_OUTFILE_NA = f'{RESULTS_DIR}/heatmap_mfd_NA_{MODEL_NAME_SAFE}.png'


# ------------------------------
# Definitions
# ------------------------------

def load_mfd_from_dic(filepath):
    """
    Parses a standard LIWC-style .dic file (used by MFD).
    """
    mfd_dict = {}
    id_to_cat = {}
    
    mode = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line == '%':
                    mode += 1
                    continue
                
                if mode == 1:
                    parts = line.split()
                    if len(parts) >= 2:
                        cat_id = parts[0]
                        cat_name = parts[1]
                        # # Combine category name (care.virtue -> care) virtue/vice
                        # clean_cat_name = cat_name.split('.')[0] 
                        # id_to_cat[cat_id] = clean_cat_name
                        
                        # Dont Combine category name (care.virtue -> care.virtue/vice)
                        id_to_cat[cat_id] = cat_name
                
                elif mode == 2:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        cat_id = parts[1] 
                        
                        if cat_id in id_to_cat:
                            mfd_dict[word] = id_to_cat[cat_id]
                            
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return {}
  
    return mfd_dict


def get_word_embedding(word_span, offsets, hidden_states):
    """Extracts and averages token embeddings for a specific character span."""
    word_start, word_end = word_span
    relevant_indices = []

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start < word_end and tok_end > word_start:
            relevant_indices.append(i)
            
    if not relevant_indices:
        return None

    selected_vectors = hidden_states[relevant_indices]
    word_embedding = torch.mean(selected_vectors, dim=0) # Shape: (768,)
    return word_embedding.detach().cpu().numpy()

def process_post_chunks(post_text, tokenizer, model, mfd_dict):
    """
    Handles long posts by sliding window. Returns a list of records:
    [{'word': 'help', 'vec': np.array, 'type': 'mfd', 'foundation': 'care'}, ...]
    """
    
    # Identify words using Regex
    # We map start_char -> word_info to quickly look up tokens later
    # Uses regex to get the boundry indexes of every word
    # lowercases the post words
    # checks if this word is also in the MFG dic, if so
    #    appends it to a "word_matches" list that keeps the word, 
    #    the start/end indexes, the foundation and type = mdf
    word_matches = []
    for match in re.finditer(r'\b\w+\b', post_text):
        word = match.group().lower()
        if word in mfd_dict:
            word_matches.append((match.span(), word, 'mfd', mfd_dict[word]))

    if not word_matches:
        return []

    # Tokenize with Overflow (Chunking)
    # stride=128 means windows overlap slightly so we don't cut words in half
    inputs = tokenizer(post_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        stride=128,
                        return_overflowing_tokens=True, 
                        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop("offset_mapping")
    overflow_mapping = inputs.pop("overflow_to_sample_mapping")
    
    # Move to GPU
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    extracted_data = []

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # Iterate over every chunk (window)
    for i, offsets in enumerate(offset_mapping):
        hidden_state = last_hidden_states[i] # (Seq_Len, 768)
        offsets = offsets.tolist() # List of (start, end)

        # Check every MFD word in the post to see if it falls in this chunk
        for (start, end), word, w_type, foundation in word_matches:
            # Check if this character span is covered by this chunk's offsets
            # The chunk covers offsets[1][0] to offsets[-2][1] (ignoring CLS/SEP)
            chunk_start_char = offsets[1][0]
            chunk_end_char = offsets[-2][1]

            #If the word does fall in this chunk, get the word embedding for it
            if start >= chunk_start_char and end <= chunk_end_char:
                #Covnert token(s) back to word and get its embedding
                vec = get_word_embedding((start, end), offsets, hidden_state)
                if vec is not None:
                    extracted_data.append({
                        'word': word,
                        'vector': vec,
                        'type': w_type,
                        'foundation': foundation
                    })
    
    return extracted_data

def generate_heatmap(results_list, title_suffix, filename):
    df = pd.DataFrame(results_list)
    
    if df.empty:
        print(f"No data found for {title_suffix}")
        return

    print(f"\n--- Generating Matrix for {title_suffix} ---")
    
    # Pivot
    pivot_matrix = df.pivot_table(
        index='topic_id', 
        columns='foundation', 
        values='shift_dist', 
        aggfunc='mean'
    )
    
    print(pivot_matrix)
    # Fill NaN with 0 (implies no shift/no data)
    #pivot_matrix = pivot_matrix.fillna(0)
    
    plt.figure(figsize=(12, 10))
    # Using 'magma' because we are measuring distance (0=black/normal, Bright=High Shift)
    #sns.heatmap(pivot_matrix, annot=True, cmap="magma", fmt=".2f")
    sns.heatmap(pivot_matrix, annot=True, cmap="bwr", fmt=".2f", center=0)
    plt.title(f"Semantic Shift: {title_suffix}")
    plt.xlabel("Moral Foundation")
    plt.ylabel("Topic ID")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

# ------------------------------
# Start timer
# ------------------------------
start_time = time.time()

# ------------------------------
# Load MFD2.0 Dictonary
# ------------------------------
mfd_dict = load_mfd_from_dic(MFD_FILE) 

# ------------------------------
# Load All Posts
# ------------------------------
raw_data = pd.read_pickle(DATA_FILE)
print(f'Read in {raw_data.shape[0]} posts')

topics = [random.randint(0, 44) for _ in range(raw_data.shape[0])]

# subset = 100
# raw_data = raw_data.iloc[:subset]
# topics = [random.randint(0, 44) for _ in range(subset)]

# # 10 Sample Posts (Assumed 1 post = 1 topic for this test)
# posts = [
#     "I want to help my mother but she treats me with cruelty.", 
#     "To kill a mockingbird is a story about justice and fairness.", 
#     "I paid for the car but he committed fraud to cheat me out of money.",
#     "We must respect our elders and obey the laws of the land.",
#     "The kindness of strangers helped me suffer less during the war.",
#     "It is not fair to hurt innocent people for no reason.",
#     "He did not show care, he only wanted to kill the mood.",
#     "I have to respect the decision even if it is a fraud.",
#     "They help the poor and seek justice for the weak.",
#     "Obey the rules or you will hurt the team spirit and soulful."
# ]
# # Assign Topic IDs (0 to 9)
# topics = list(range(10))

# ------------------------------
# Load model & tokenizer
# ------------------------------
# tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
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


# Storage
# global_registry: stores ALL vectors for a word (to calc global baseline)
# topic_registry: stores vectors per topic
global_registry = defaultdict(list)
topic_registry = defaultdict(lambda: defaultdict(list))

print(f"Processing {len(raw_data)} posts...")

for topic_id, (index, row) in tqdm(zip(topics, raw_data.iterrows()), total=len(raw_data), desc="Getting embeddings for MFD per post"):
    # Extract vectors
    post_text = row['selftext']
    post_id = index
    # Unique values in 'resolved_verdict':
    # ['YTA' 'NTA' 'ESH' 'NAH']
    verdict = row['resolved_verdict']
    if verdict in ["YTA", "ESH"]:
        verdict = "YA"
    else:
        verdict = "NA"

    records = process_post_chunks(post_text, tokenizer, model, mfd_dict)

    for rec in records:
        w = rec['word']
        v = rec['vector']
        
        # Add to Global List
        global_registry[w].append(v)
        
        # Add to Topic List
        #topic_registry[topic_id][w].append(v)
        topic_registry[topic_id][w].append( (v, post_id, verdict) )

print("Processing complete. Calculating Shifts...")

# ------------------------------
# 4. Calculation Phase
# ------------------------------

# A. Compute Global Centroids of every MFD word's vectors
# output is "word:average vector"
# len(global_registry[word]) will give how many times
#    that word was found in the global corpus
global_centroids = {}
for word, vectors in tqdm(global_registry.items(), desc="Processing global baseline vectors"):
    global_centroids[word] = np.mean(vectors, axis=0)

# print("\nWord Counts:")
# for word, vectors in global_registry.items():
#     print(f"{word}: {len(vectors)}")

# B. Compute Moral Shift per Word per Topic
# Loop throgh all the MFD words in a topic and get 
# the cossim from that same word from the whole 100k
# docs.
results_ya = []
results_na = []

for topic_id, words_in_topic in tqdm(topic_registry.items(), desc="Calcualting shift per topic"):
    for word, data_list in words_in_topic.items():

        # data_list contains tuples: [(vec, pid, 'YA'), (vec, pid, 'NA'), ...]
        vectors_ya = [item[0] for item in data_list if item[2] == 'YA']
        vectors_na = [item[0] for item in data_list if item[2] == 'NA']

        global_vec = global_centroids[word].reshape(1, -1) # get the global average of that word accross the corpus

        if vectors_ya:
            topic_vec_ya = np.mean(vectors_ya, axis=0).reshape(1, -1)
            dist_ya = 1 - cosine_similarity(topic_vec_ya, global_vec)[0][0]
            
            results_ya.append({
                'topic_id': topic_id,
                'word': word,
                'foundation': mfd_dict.get(word, "Unknown"),
                'shift_dist': dist_ya,
                'count': len(vectors_ya)
            })
            
        if vectors_na:
            topic_vec_na = np.mean(vectors_na, axis=0).reshape(1, -1)
            dist_na = 1 - cosine_similarity(topic_vec_na, global_vec)[0][0]
            
            results_na.append({
                'topic_id': topic_id,
                'word': word,
                'foundation': mfd_dict.get(word, "Unknown"),
                'shift_dist': dist_na,
                'count': len(vectors_na)
            })



# ------------------------------
# 5. Report: Foundation Shifts per Topic
# ------------------------------
generate_heatmap(results_ya, "You're the Asshole (YA)", HEATMAP_OUTFILE_YA)
generate_heatmap(results_na, "Not the Asshole (NA)", HEATMAP_OUTFILE_NA)


# ------------------------------
# End timer & report
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"Total execution time: {mins:.0f} min {secs:.2f} sec")



