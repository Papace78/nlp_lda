import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

all_vectorizer = TfidfVectorizer()

all_vectorized = all_vectorizer.fit_transform(all_interviews)
all_vectorized = pd.DataFrame(all_vectorized.toarray(),columns = all_vectorizer.get_feature_names_out())


n_components = 4
lda_model = LatentDirichletAllocation(n_components=n_components, max_iter = 100)

lda_model.fit(all_vectorized)

def print_topics(lda_model, vectorizer, top_words):
    # 1. TOPIC MIXTURE OF WORDS FOR EACH TOPIC
    topic_mixture = pd.DataFrame(
        lda_model.components_,
        columns = vectorizer.get_feature_names_out()
    )

    # 2. FINDING THE TOP WORDS FOR EACH TOPIC
    ## Number of topics
    n_components = topic_mixture.shape[0]

    ## Top words for each topic
    for topic in range(n_components):
        print("-"*10)
        print(f"For topic {topic}, here are the the top {top_words} words with weights:")

        topic_df = topic_mixture.iloc[topic]\
            .sort_values(ascending = False).head(top_words)

        print(round(topic_df,3))
