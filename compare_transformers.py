import os
import time

import numpy as np
import pandas as pd
from bertopic import BERTopic

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# ------------------------------
# Config
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATA_FILE = os.path.join(DATA_DIR, "aita_filtered.pkl")

# MPNet paths
MPNET_MODEL_FILE = os.path.join(DATA_DIR, "bertopic_model_mpnet")
MPNET_EMB_FILE = os.path.join(DATA_DIR, "doc_embeddings_mpnet_base_fp16.npy")

# RoBERTa paths
ROBERTA_MODEL_FILE = os.path.join(DATA_DIR, "bertopic_model_roberta_final")
ROBERTA_EMB_FILE = os.path.join(DATA_DIR, "doc_embeddings_roberta_base.npy")


# ------------------------------
# Definitions
# ------------------------------
def load_docs():
    print(f"Loading data from {DATA_FILE} ...")
    df = pd.read_pickle(DATA_FILE)
    docs = df["selftext"].fillna("").tolist()
    return docs


def load_embeddings(path: str):
    print(f"Loading embeddings from {path} ...")
    embeddings = np.load(path)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings


# From Google Overview
def save_plotly_png(fig, png_path: str):
    fig.write_image(png_path)
    print(f"Saved figure to {png_path}")


def compute_topic_words(topic_model, top_n: int = 10):
    """Return a list of topic word lists, excluding topic -1."""
    topic_info = topic_model.get_topic_info()
    topic_ids = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

    # From ChatGPT
    topics_words = []
    for tid in topic_ids:
        words_scores = topic_model.get_topic(tid)[:top_n]
        words = [w for (w, s) in words_scores]
        topics_words.append(words)
    return topics_words, topic_ids

# From Google Overview
def compute_coherence(docs, topics_words, coherence_type="c_v", max_docs=20000):
    """
    Compute topic coherence using gensim.
    """
    n_docs = len(docs)
    if max_docs is not None and n_docs > max_docs:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_docs, size=max_docs, replace=False)
        idx = np.sort(idx)
        docs_sampled = [docs[i] for i in idx]
    else:
        docs_sampled = docs

    # Whitespace tokenization
    texts = [d.lower().split() for d in docs_sampled]

    dictionary = Dictionary(texts)
    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type,
    )
    coherence = coherence_model.get_coherence()
    return coherence


def evaluate_model(
    model_name: str,
    topic_model: BERTopic,
    docs,
    embeddings,
):
    print(f"\nQuantitative evaluation for {model_name}")

    # Topic words
    topics_words, topic_ids = compute_topic_words(topic_model, top_n=10)
    num_topics = len(topic_ids)
    print(f"Number of topics (excluding -1): {num_topics}")

    # Topic coherence (c_v)
    coherence_cv = compute_coherence(docs, topics_words, coherence_type="c_v", max_docs=20000)
    print(f"Topic Coherence (c_v): {coherence_cv:.4f}")

    print(f"----- Done evaluating {model_name} -----\n")


def visualize_model(
    model_name: str,
    model_path: str,
    embeddings_path: str,
    docs,
    out_dir: str,
    sample_fraction: float = 0.25,
):
    print(f"\nVisualizing model: {model_name}")

    print("Loading BERTopic model ...")
    topic_model = BERTopic.load(model_path)

    # Quantitative evaluation
    embeddings_full = load_embeddings(embeddings_path)
    evaluate_model(model_name, topic_model, docs, embeddings_full)

    # Topic naming by Henry Arze (MPNet only)
    TOPIC_LABELS = {
        0:  "dating",
        1:  "communication",
        2:  "employment",
        3:  "apologies",
        4:  "chores",
        5:  "nighttime",
        6:  "school",
        7:  "morning",
        8:  "marriage",
        9:  "driving",
        10: "pets (adoption)",
        11: "flight",
        12: "dining",
        13: "housing",
        14: "close relationship (strain)",
        15: "relationship",
        16: "engagement/wedding",
        17: "video games",
        18: "affairs",
        19: "transactions",
        20: "roommates",
        21: "pregnancy",
        22: "pets (ownership)",
        23: "expectations",
        24: "confrontation",
        25: "rent",
        26: "babies",
        27: "bills",
        28: "vacation/holiday",
        29: "food",
        30: "coworkers",
        31: "parking",
        32: "finances",
        33: "religion",
        34: "physical health",
        35: "neighbors",
        36: "workplace",
        37: "expression",
        38: "cheating",
        39: "depression",
        40: "family problems",
        41: "cooking",
        42: "drinking",
        43: "gifts",
        44: "romance",
        45: "scheduling",
        46: "mental health",
        47: "community",
    }

    # Use custom labels
    if model_name == "mpnet":
        topic_model.set_topic_labels(TOPIC_LABELS)

    # Bar chart: topic-word distributions
    fig_bar = topic_model.visualize_barchart(width=280, height =330, top_n_topics=8, n_words = 10, custom_labels=True)
    save_plotly_png(fig_bar, os.path.join(out_dir, f"{model_name}_barchart.png"))

    # Documents scatter plot
    embeddings = load_embeddings(embeddings_path)

    fig_docs = topic_model.visualize_documents(
        docs,
        embeddings=embeddings,
        sample=sample_fraction,  # fraction of docs to include in the plot
        custom_labels=True,
    )
    save_plotly_png(fig_docs, os.path.join(out_dir, f"{model_name}_documents.png"))

    print(f"Finished visualizations for {model_name}.\n")


def main():

    docs = load_docs()

    # MPNet visualizations
    mpnet_results_dir = os.path.join(RESULTS_DIR, "mpnet")
    visualize_model(
        model_name="mpnet",
        model_path=MPNET_MODEL_FILE,
        embeddings_path=MPNET_EMB_FILE,
        docs=docs,
        out_dir=mpnet_results_dir,
        sample_fraction=0.25,  # ~25% of 108k
    )

    # RoBERTa visualizations
    roberta_results_dir = os.path.join(RESULTS_DIR, "roberta")
    visualize_model(
        model_name="roberta",
        model_path=ROBERTA_MODEL_FILE,
        embeddings_path=ROBERTA_EMB_FILE,
        docs=docs,
        out_dir=roberta_results_dir,
        sample_fraction=0.25,
    )


if __name__ == "__main__":
    main()
