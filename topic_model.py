import os
import time

import numpy as np
import pandas as pd

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_FILE = os.path.join(DATA_DIR, "aita_filtered.pkl")
# EMB_FILE = os.path.join(DATA_DIR, "doc_embeddings_mpnet_base_fp16.npy")
EMB_FILE = os.path.join(DATA_DIR, "doc_embeddings_roberta_base.npy")
TOPIC_MODEL_FILE = os.path.join(DATA_DIR, "bertopic_model_roberta_final")
TOPIC_ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "final_aita_with_topics_roberta.csv")

TOP_WORDS_TXT_FILE = os.path.join(DATA_DIR, "roberta_top20_words.txt")

def main():

    # Load data + embeddings
    print(f"Loading data from {DATA_FILE} ...")
    df = pd.read_pickle(DATA_FILE)
    docs = df["selftext"].fillna("").tolist()
    print(f"Loaded {len(docs)} documents.")

    print(f"Loading embeddings from {EMB_FILE} ...")
    embeddings = np.load(EMB_FILE)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Create BERTopic model

    # Ensure representative words are useful
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )

    print("Fitting BERTopic model...")
    # Dim Reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # Non-parametric clustering (no K)
    hdbscan_model = HDBSCAN(
        min_cluster_size=100,     # ~minimum topic size; tweak this, try 300
        min_samples=5,           # higher = stricter, fewer clusters, try 10
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    # Topic modeling using BERTopic
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # ------------------------------
    # Add topics label, topic pairs, and top 2 topic probabilities to df
    # ------------------------------
    df = df.copy()
    topics = np.array(topics)
    df["topic_id_hard"] = topics  # raw BERTopic labels (can be -1)

    probs = np.array(probs)

    # Map probability columns to actual topic IDs
    topic_info = topic_model.get_topic_info()
    # Exclude outlier topic (-1)
    valid_topic_rows = topic_info[topic_info["Topic"] != -1]
    topic_id_order = valid_topic_rows["Topic"].tolist()  # length K

    K = probs.shape[1]

    # Sort topic indices by probability (most probable to least)
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]  # [N, K]

    # Save Topic IDS to df
    top1_idx = sorted_idx[:, 0]
    top1_ids = [topic_id_order[j] for j in top1_idx]
    df["topic_id_top1"] = top1_ids

    top2_idx = sorted_idx[:, 1]
    top2_ids = [topic_id_order[j] for j in top2_idx]
    df["topic_id_top2"] = top2_ids

    # Include probablity of top 2 topics
    row_idx = np.arange(len(df))
    df["topic_prob_top1"] = probs[row_idx, top1_idx]
    df["topic_prob_top2"] = probs[row_idx, top2_idx]

    # --------------------------
    # Create unordered topic pair
    # --------------------------
    def make_topic_pair(row):
        t1 = row["topic_id_top1"]
        t2 = row["topic_id_top2"]
        a, b = sorted([int(t1), int(t2)])
        return f"[{a},{b}]"

    df["topic_pair"] = df.apply(make_topic_pair, axis=1)

    # Save work
    print(f"Saving BERTopic model to {TOPIC_MODEL_FILE} ...")
    topic_model.save(TOPIC_MODEL_FILE)

    print(f"Saving topic pair CSV to {TOPIC_ASSIGNMENTS_FILE} ...")
    df.to_csv(TOPIC_ASSIGNMENTS_FILE, index=False)

    # Short topic overview
    topic_info = topic_model.get_topic_info()
    print("\nTop 10 topics:")
    print(topic_info.head(11))
    print("n_topics:", topic_info["Topic"].nunique())

    # ChatGPT
    # Write top 20 words per topic to a text file
    print(f"Writing top 20 words per topic to {TOP_WORDS_TXT_FILE} ...")
    with open(TOP_WORDS_TXT_FILE, "w", encoding="utf-8") as f:
        for topic_id in topic_info["Topic"].tolist():
            if topic_id == -1:
                continue  # skip outlier topic

            top_words_scores = topic_model.get_topic(topic_id)[:20]
            word_list = [w for (w, s) in top_words_scores]

            f.write(f"Topic {topic_id}:\n")
            f.write("  " + ", ".join(word_list) + "\n\n")

    print("Done.")


if __name__ == "__main__":
    main()
