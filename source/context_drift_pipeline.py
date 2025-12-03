import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os
import spacy


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

MODEL_NAME = "roberta-base" # 15min on H200, 53 min w NER
# MODEL_NAME = "microsoft/mpnet-base" # 20min on H200, 60 min w NER
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# MODEL_NAME = "allenai/longformer-base-4096"
MODEL_NAME_SAFE = MODEL_NAME.replace("/", "-")
MAX_LEN = 512       # 512 for roberta,mpnet, 4096 for longformer
STRIDE = 128         # 128 for roberta,mpnet, 512 for longformer

HEATMAP_OUTFILE_YA = f'{RESULTS_DIR}/heatmap_mfd_YA_{MODEL_NAME_SAFE}.png'
HEATMAP_OUTFILE_NA = f'{RESULTS_DIR}/heatmap_mfd_NA_{MODEL_NAME_SAFE}.png'


# ------------------------------
# Definitions
# ------------------------------

def get_ner_indices(text):
    """
    Returns a set of character indices that belong to Named Entities.
    """
    doc = nlp(text)
    forbidden_indices = set()
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON", "FAC", "LOC", "DATE", "EVENT"]:
            for i in range(ent.start_char, ent.end_char):
                forbidden_indices.add(i)
                
    return forbidden_indices

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
    word_embedding = torch.mean(selected_vectors, dim=0)
    return word_embedding.detach().cpu().numpy()

def process_post_chunks(post_text, tokenizer, model, mfd_dict):
    """
    Handles long posts by sliding window. Returns a list of records:
    [{'word': 'help', 'vec': np.array, 'type': 'mfd', 'foundation': 'care'}, ...]
    """
    
    # Identify words using Regex
    # Map start_char -> word_info to quickly look up tokens later
    # Uses regex to get the boundry indexes of every word
    # lowercases the post words
    # checks if this word is also in the MFG dic, if so
    #    appends it to a "word_matches" list that keeps the word, 
    #    the start/end indexes, the foundation and type = mdf
    # and ignore named entities

    ner_map = get_ner_indices(post_text)

    word_matches = []
    for match in re.finditer(r'\b\w+\b', post_text):
        word = match.group().lower()
        start_char, end_char = match.span()
    
        if start_char in ner_map:
            continue
        
        if word in mfd_dict:
            word_matches.append((match.span(), word, 'mfd', mfd_dict[word]))

    if not word_matches:
        return []

    inputs = tokenizer(post_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=MAX_LEN, 
                        stride=STRIDE,
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

    for i, offsets in enumerate(offset_mapping):
        hidden_state = last_hidden_states[i]
        offsets = offsets.tolist()


        for (start, end), word, w_type, foundation in word_matches:
            chunk_start_char = offsets[1][0]
            chunk_end_char = offsets[-2][1]

            if start >= chunk_start_char and end <= chunk_end_char:
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

    pivot_matrix = df.pivot_table(
        index='topic_id', 
        columns='foundation', 
        values='shift_dist', 
        aggfunc=np.max
    )
    
    print(pivot_matrix)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_matrix, annot=True, cmap="bwr", fmt=".2f", center=0)
    plt.title(f"Semantic Shift: {title_suffix}")
    plt.xlabel("Moral Foundation")
    plt.ylabel("Topic Pair")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

# ------------------------------
# Start timer
# ------------------------------
start_time = time.time()

# ------------------------------
# Load NER and MFD2.0 Dictonary
# ------------------------------
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

mfd_dict = load_mfd_from_dic(MFD_FILE) 

# ------------------------------
# Load All Posts
# ------------------------------
raw_data = pd.read_pickle(DATA_FILE)
print(f'Read in {raw_data.shape[0]} posts')

topics_csv = pd.read_csv(f"{DATA_DIR}/aita_with_topics.csv")
print("The Topics looks like:")
print(topics_csv.head())
print(topics_csv.shape)
print(topics_csv.info())
print(topics_csv['topic_pair'].value_counts())
print(topics_csv['topic_pair'].value_counts().head(20))
print(topics_csv.apply(pd.Series.nunique))

#Enumarate topic pairs column for plotting
pair_counts = topics_csv['topic_pair'].value_counts()
unique_pairs = pair_counts.index.tolist()
pair_to_id = {pair: i for i, pair in enumerate(unique_pairs)}
topics_csv['topic_pair_id'] = topics_csv['topic_pair'].map(pair_to_id)
topics = topics_csv['topic_pair_id']


# ------------------------------
# Load model & tokenizer
# ------------------------------
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


global_registry = defaultdict(list)
topic_registry = defaultdict(lambda: defaultdict(list))

print(f"Processing {len(raw_data)} posts...")

for topic_id, (index, row) in tqdm(zip(topics, raw_data.iterrows()), total=len(raw_data), desc="Getting embeddings for MFD per post"):
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

        global_registry[w].append(v)        
        topic_registry[topic_id][w].append( (v, post_id, verdict) )

print("Processing complete. Calculating Shifts...")

# ------------------------------
# Compute Global Centroids
#
# output is "word:average vector"
# len(global_registry[word]) will give how many times
#    that word was found in the global corpus
# ------------------------------

global_centroids = {}
for word, vectors in tqdm(global_registry.items(), desc="Processing global baseline vectors"):
    global_centroids[word] = np.mean(vectors, axis=0)

# ------------------------------
# Compute Moral Shift per MFD wor
#  per Topic
#
# Loop throgh all the MFD words in a topic and get 
# the cossim from that same word from the whole 100k
# docs.
# ------------------------------

id_to_pair = {v: k for k, v in pair_to_id.items()}
results_ya = []
results_na = []

for topic_id, words_in_topic in tqdm(topic_registry.items(), desc="Calcualting shift per topic"):
    if topic_id == -1 or topic_id > 20: 
        continue
    for word, data_list in words_in_topic.items():

        vectors_ya = [item[0] for item in data_list if item[2] == 'YA']
        vectors_na = [item[0] for item in data_list if item[2] == 'NA']

        global_vec = global_centroids[word].reshape(1, -1)
        topic_pair_name = id_to_pair[topic_id]

        if vectors_ya:
            topic_vec_ya = np.mean(vectors_ya, axis=0).reshape(1, -1)
            dist_ya = 1 - cosine_similarity(topic_vec_ya, global_vec)[0][0]
            
            results_ya.append({
                'topic_id': topic_pair_name,
                'word': word,
                'foundation': mfd_dict.get(word, "Unknown"),
                'shift_dist': dist_ya,
                'count': len(vectors_ya)
            })
            
        if vectors_na:
            topic_vec_na = np.mean(vectors_na, axis=0).reshape(1, -1)
            dist_na = 1 - cosine_similarity(topic_vec_na, global_vec)[0][0]
            
            results_na.append({
                'topic_id': topic_pair_name,
                'word': word,
                'foundation': mfd_dict.get(word, "Unknown"),
                'shift_dist': dist_na,
                'count': len(vectors_na)
            })


# ------------------------------
# Save Results
# ------------------------------
print("Saving results to CSV...")

df_ya = pd.DataFrame(results_ya)
df_na = pd.DataFrame(results_na)

df_ya.to_csv(f"{DATA_DIR}/semantic_shift_results_YA_{MODEL_NAME_SAFE}.csv", index=False)
df_na.to_csv(f"{DATA_DIR}/semantic_shift_results_NA_{MODEL_NAME_SAFE}.csv", index=False)

print(f"Saved semantic shift results to {DATA_DIR}")

# ------------------------------
# Create Heatmap for Foundation 
#  Shifts per Topic
# ------------------------------
generate_heatmap(results_ya, "You're the A (YA)", HEATMAP_OUTFILE_YA)
generate_heatmap(results_na, "Not the A (NA)", HEATMAP_OUTFILE_NA)

# ------------------------------
# End timer & report
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"Total execution time: {mins:.0f} min {secs:.2f} sec")



