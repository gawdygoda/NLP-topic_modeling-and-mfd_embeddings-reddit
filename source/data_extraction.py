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
            "_id": None,
            "max_num_words": {"$max": "$num_words"}
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
            "_id": None,
            "max_num_words": {"$max": "$num_tokens"}
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
    "_id": 0,
    "title": 1,
    "author": 1,
    "selftext": 1,
    "num_comments": 1,
    "created_utc": 1,
    "num_words": 1,
    "resolved_verdict": 1,
}

# docs = list(subs.find(filter_criteria, projection))

# df = pd.DataFrame(docs)

# # Save DataFrame to pickle
# df.to_pickle(DATA_FILE)

#print(subs.find_one())


# Total Posts: 148691
# Max num_words:
#  [{'_id': None, 'max_num_words': 6729}]
# Max num_tokens:
#  [{'_id': None, 'max_num_words': 6729}]
# Number of posts with more than 512 words: 38125
# Number of documents with no resolved_verdict: 0
# resolved_verdict:
# {'_id': '', 'count': 920}
# {'_id': 'YTA', 'count': 31959}
# {'_id': 'NTA', 'count': 83535}
# {'_id': 'INFO', 'count': 4224}
# {'_id': 'ESH', 'count': 9359}
# {'_id': 'NAH', 'count': 18694}
# Number of documents matching filter: 108634
