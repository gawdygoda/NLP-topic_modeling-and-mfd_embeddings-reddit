import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------------------
# Config
# ------------------------------
# MODEL_NAME = "roberta-base"
MODEL_NAME = "microsoft/mpnet-base"
MODEL_NAME_SAFE = MODEL_NAME.replace("/", "-")
DATA_DIR = "../data"
RESULTS_DIR = "../results"
FILE_PATH = f"{DATA_DIR}/semantic_shift_results_YA_{MODEL_NAME_SAFE}.csv" # Or NA
OUTFILE_PLOT = f'{RESULTS_DIR}/semantic_shift_results_YA_{MODEL_NAME_SAFE}.png'


# ------------------------------
# Definitions
# ------------------------------

def plot_top_shifting_words(file_path, target_topic, target_foundation, top_n=10):
    """
    Plots bar chart of the words with the highest shift
    """
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    subset = df[
        (df['topic_id'] == target_topic) & 
        (df['foundation'] == target_foundation)
    ].copy()
    
    if subset.empty:
        print(f"No data found for Topic: {target_topic} and Foundation: {target_foundation}")
        print("Available Topics:", df['topic_id'].unique()[:10])
        print("Available Foundations:", df['foundation'].unique())
        return

    subset = subset.sort_values(by='shift_dist', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=subset,
        x='shift_dist',
        y='word',
        palette='magma'
    )
    
    plt.title(f"Top {top_n} Shifting Words\nTopic: {target_topic} | Foundation: {target_foundation}")
    plt.xlabel("Cosine Shift Distance (Higher = More Meaning Change)")
    plt.ylabel("Word")
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig(OUTFILE_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTFILE_PLOT}")

# ------------------------------
# Make Plot
# ------------------------------

MY_TOPIC = "[15,18]"
# "fairness.vice", "authority.virtue"
MY_FOUNDATION = "authority.virtue"

plot_top_shifting_words(FILE_PATH, MY_TOPIC, MY_FOUNDATION, top_n=10)
