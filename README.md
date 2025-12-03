# Topic Modeling and Insights from AITA Subreddit

Linguistic analysis with topic modeling and transformer based embeddings to discover moral foundation themes in reddit threads

This project is modeled off of Josh Nguyen's PhD Thesis

https://joshnguyen.net/files/JoshNguyen_MPhil_Thesis.pdf

The dataset and code can be found at https://github.com/joshnguyen99/moral_dilemma_topics

The general process is:

1. [Data Extraction](./source/data_extraction.py)
2. [Preprocessing and EDA](./source/preprocessing_eda.py)
3. [Topic Modeling](./source/topic_modeling_pipeline.py)
4. [Context Drift Analysis](./source/context_drift_pipeline.py)
5. [Results & Plotting](./source/results_ploting.py)

---

## Required Dependencies
> pip install pymongo transformers  scikit-learn seaborn bertopic spacy

> python -m spacy download en_core_web_sm



