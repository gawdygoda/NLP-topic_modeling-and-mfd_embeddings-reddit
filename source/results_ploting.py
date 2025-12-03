import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------------------
# Config
# ------------------------------
# MODEL_NAME = "roberta-base" # 15min on H200
MODEL_NAME = "microsoft/mpnet-base" # 20min on H200
MODEL_NAME_SAFE = MODEL_NAME.replace("/", "-")
DATA_DIR = "../data"
RESULTS_DIR = "../results"
FILE_PATH = f"{DATA_DIR}/semantic_shift_results_YA_{MODEL_NAME_SAFE}.csv" # Or NA
OUTFILE_PLOT = f'{RESULTS_DIR}/semantic_shift_results_YA_{MODEL_NAME_SAFE}.png'

def plot_top_shifting_words(file_path, target_topic, target_foundation, top_n=10):
    """
    Plots a horizontal bar chart of the words with the highest semantic shift
    within a specific Topic and Moral Foundation.
    """
    
    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # 2. Filter for the specific Topic and Foundation
    # Ensure target_topic matches the string format in your CSV (e.g., "[3,13]")
    subset = df[
        (df['topic_id'] == target_topic) & 
        (df['foundation'] == target_foundation)
    ].copy()
    
    if subset.empty:
        print(f"No data found for Topic: {target_topic} and Foundation: {target_foundation}")
        print("Available Topics:", df['topic_id'].unique()[:10]) # Show sample
        print("Available Foundations:", df['foundation'].unique())
        return

    # 3. Sort by Shift Distance (Descending) and take Top N
    subset = subset.sort_values(by='shift_dist', ascending=False).head(top_n)
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Create barplot
    ax = sns.barplot(
        data=subset,
        x='shift_dist',
        y='word',
        palette='magma'
    )
    
    # Formatting
    plt.title(f"Top {top_n} Shifting Words\nTopic: {target_topic} | Foundation: {target_foundation}")
    plt.xlabel("Cosine Shift Distance (Higher = More Meaning Change)")
    plt.ylabel("Word")
    
    # Optional: Add text labels to bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig(OUTFILE_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTFILE_PLOT}")

# ------------------------------
# Example Usage
# ------------------------------

# Change these to the Topic String and Foundation you want to inspect
MY_TOPIC = "[15,18]"  # Use the exact string format from your ID map
MY_FOUNDATION = "authority.virtue" # Or "fairness.vice", "authority.virtue", etc.

plot_top_shifting_words(FILE_PATH, MY_TOPIC, MY_FOUNDATION, top_n=10)
