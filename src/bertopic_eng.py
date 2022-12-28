# Importing dependencies
import os
import pandas as pd
import random
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from viz_functions import *

random.seed(29)

# loading the data
path_pkl = os.path.join("data", "full_df.pkl")
data = pd.read_pickle(path_pkl) 

data = data[data["confidence"] > 0.9]

# creating subsets
eng_data = data[data["language"] == "en"]

# finding the number of days
days = eng_data['date'].unique()
print('The number of days is: ' + str(len(days)))

# the number of documents to be sampled per day
n_sample = round(300000/len(days))

full_df = eng_data[eng_data.date == days[0]].sample(n = n_sample, axis = 0, random_state=29)
for i in range(len(days)-1):
    eng_data_day = eng_data[eng_data.date == days[i+1]]
    eng_data_day = eng_data_day.sample(n = n_sample, axis = 0, random_state=29)
    full_df = pd.concat([full_df, eng_data_day], ignore_index=True)

print(full_df.shape)

# make text column lower case
full_df.document = full_df['document'].map(lambda document: document.lower() if isinstance(document,str) else document)

# make lists
dates = full_df.date.to_list()
docs = full_df.document.to_list()
subreddits = full_df.sub_reddit.to_list()
language = full_df.language.to_list()
type = full_df.type.to_list()

#If this outputs anything, there is a problem. Something that should be a string is not...
for i in docs: 
    test = isinstance(i, str)
    if test == False: 
        print(test)

# Prepare embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = sentence_model.encode(docs, show_progress_bar=True)
path_save_emb = os.path.join("out", "embedding_eng_downsampled")
np.save(path_save_emb, embeddings)

# Fitting the model
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state = 29)

# Bertopic
topic_model = BERTopic(language = "multilingual", verbose = True, calculate_probabilities = False, umap_model=umap_model, nr_topics='auto', low_memory=True)

topics, probs = topic_model.fit_transform(docs, embeddings)

# Removing stopwords
english_stopwords = get_stop_words('en')

stopwords = list(english_stopwords)

vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stopwords)

topic_model.update_topics(docs, vectorizer_model=vectorizer_model)

# Save and load model

path_save_model = os.path.join("out", "eng_model_downsampled")
topic_model.save(path_save_model)
topic_model = BERTopic.load(path_save_model)

path_save_1 = os.path.join("out", "topics_over_time_eng_downsampled.html")
path_save_2 = os.path.join("out", "topics_over_time_class_eng_downsampled.html")

# Topics over time
topics_over_time = topic_model.topics_over_time(docs, dates, nr_bins=20)

fig1 = viz_topics_over_time(topic_model, topics_over_time, top_n_topics=20)
fig1.write_html(path_save_1)

# Topics per class
language = full_df.language.to_list()
type = full_df.type.to_list()

topics_per_class = topic_model.topics_per_class(docs, classes = language)

fig2 = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
fig2.write_html(path_save_2)









