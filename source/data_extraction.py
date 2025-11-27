from pymongo import MongoClient
import pandas as pd

DATA_DIR = "../data"
DATA_FILE = f"{DATA_DIR}/aita_filtered.pkl"

# ------------------------------
# Connect
# ------------------------------
uri = "mongodb://admin:password@localhost:27017/reddit?authSource=admin"
client = MongoClient(uri)

db = client.get_database()
subs = db.get_collection("submissions")

# ------------------------------
# Count docs
# ------------------------------
count = subs.count_documents({})
print("Total Posts:", count)


# ------------------------------
# Max Words
# ------------------------------
pipeline = [
    {"$group": {
            "_id": None,               # we don’t care about grouping by a specific field
            "max_num_words": {"$max": "$num_words"}  # compute max of num_words
        }
    }
]

result = list(subs.aggregate(pipeline))
print("Max num_words:\n", result)

# ------------------------------
# Max Tokens
# ------------------------------
pipeline = [
    {"$group": {
            "_id": None,               # we don’t care about grouping by a specific field
            "max_num_words": {"$max": "$num_tokens"}  # compute max of num_words
        }
    }
]

result = list(subs.aggregate(pipeline))
print("Max num_tokens:\n", result)

# ------------------------------
# Docs with more then 512 words (BERT Token Limit)
# ------------------------------
threshold = 512
count = subs.count_documents({"num_words": {"$gt": threshold}})
print(f"Number of posts with more than {threshold} words:", count)

# ------------------------------
# Count docs with no resolved verdict filter
# ------------------------------
filter_criteria = {
    "resolved_verdict": {"$exists": False},
}

count = subs.count_documents(filter_criteria)
print(f"Number of documents with no resolved_verdict: {count}")

# ------------------------------
# Resolved verdics Count
# ------------------------------
pipeline = [
    {"$group": {
            "_id": { "$ifNull": ["$resolved_verdict", "MISSING"] },
            "count": { "$sum": 1 }
        }
    }
]

result = list(subs.aggregate(pipeline))
print("resolved_verdict:")
for item in result:
    print(item)

# # ------------------------------
# # split Count - ignore and create our own split
# # ------------------------------
# pipeline = [
#     {"$group": {
#             "_id": { "$ifNull": ["$split", "MISSING"] },
#             "count": { "$sum": 1 }
#         }
#     }
# ]

# result = list(subs.aggregate(pipeline))
# print("split:")
# for item in result:
#     print(item)



# ------------------------------
# Count and get docs with filter
# ------------------------------
filter_criteria = {
    "num_words": {"$gte": 50, "$exists": True},
    "num_comments": {"$gte": 10, "$exists": True},
    "resolved_verdict": {
        "$exists": True,
        "$nin": [None, "", "INFO"]
    },
    "title": {"$regex": r"^(AITA|WIBTA)", "$options": "i", "$exists": True},
    "score": {"$gte": 1, "$exists": True}#,
    #"filtered": {"$ne": "false"},
    #"split": {"$in": ["train", "valid"]}
}

count = subs.count_documents(filter_criteria)
print(f"Number of documents matching filter: {count}")

projection = {
    "_id": 0,                 # remove internal MongoDB id
    "title": 1,
    "author": 1,
    "selftext": 1,
    "num_comments": 1,
    "created_utc": 1,
    "num_words": 1,
    "resolved_verdict": 1,
}

docs = list(subs.find(filter_criteria, projection))

df = pd.DataFrame(docs)

# Save DataFrame to pickle
df.to_pickle(DATA_FILE)

#print(subs.find_one())



