import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/aita_filtered.pkl"
RESULTS_DIR = "../results"


# ------------------------------
# Start Time
# ------------------------------
start_time = time.time()

# ------------------------------
# Read in dataframe
# ------------------------------
raw_data = pd.read_pickle(DATA_FILE)

# ------------------------------
# Perform EDA
# ------------------------------
print("The Dataset looks like:")
print(raw_data.head())
print(raw_data.shape)
print("====================================")

# Set options to show all columns of the dataset
pd.set_option("display.max_columns", None)
# Display all the columns together in the console
print("Display first 5 rows")
print(raw_data.head().to_string())
print("====================================")
print("Basic Dataframe info")
print(raw_data.info())
print("====================================")
print("More detailed Dataframe info")
print(raw_data.describe(include="all").to_string())
print("====================================")
print("Number of Empty values in each column:")
print(raw_data.isnull().sum().sort_values(ascending = False))
print("====================================")
print("Number of Unique values in each column:")
print(raw_data.apply(pd.Series.nunique))
print("====================================")
print("Are there duplicate rows?")
print(raw_data.duplicated().sum())
dup_all = raw_data[raw_data.duplicated(keep=False)]
print(dup_all)
print("====================================")
print("Unique values in 'resolved_verdicts':")
rv_counts = raw_data["resolved_verdict"].value_counts()
print(rv_counts)
print("====================================")
num_words = raw_data["num_words"]
max_words = raw_data["num_words"].max()
over_512 = (raw_data["num_words"] > 512).sum()
print("Max num_words:", max_words)
print("Count of rows where num_words > 512:", over_512)
print("====================================")

# ------------------------------
# Plotting
# ------------------------------

#Plot Resolved Verdict Pie Graph
plt.figure(figsize=(6,6))
plt.pie(rv_counts.values, labels=rv_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Resolved Verdicts")
filename = f'{RESULTS_DIR}/ResolvedVerdictsDistribution.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {filename}")


# Plot Num Words Histogram
plt.figure(figsize=(10, 6))
sns.histplot(num_words, bins=60, kde=False)
plt.axvline(512, color='red', linestyle='--', linewidth=2, label="~512-token limit")
percentage = over_512/len(raw_data)*100
plt.text(
    512 + 20,
    plt.ylim()[1] * 0.9,
    f"{over_512:,}({percentage:.0f}%) posts > 512 words",
    color='red',
    fontsize=12
)
plt.title(f"Histogram of Post Word Counts\nPost with Max words = {max_words:,}\nNote: Word count is a rough proxy for BERT’s token limit")
plt.xscale("log")
plt.xlabel("Number of Words")
plt.ylabel("Number of Posts")
plt.legend()
plt.tight_layout()
filename = f'{RESULTS_DIR}/NumWordsHistogram.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {filename}")

# ------------------------------
# End Time
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

print(f"Total time: {mins:.0f} min {secs:.2f} sec")
