import pandas as pd
import time

DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/aita_filtered.pkl"

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

# ------------------------------
# End Time
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

print(f"Total time: {mins:.0f} min {secs:.2f} sec")