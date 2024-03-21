import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV


def tfid_vectorizer(preproc_texts: pd.DataFrame, ngram_range: tuple = (1,1)) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    vectorized_texts = vectorizer.fit_transform(preproc_texts)
    vectorized_texts = pd.DataFrame(vectorized_texts.toarray(),columns = vectorizer.get_feature_names_out())

    return vectorizer, vectorized_texts


def lda(vectorized_texts, n_components: int = 2, max_iter: int = 100):

    lda_model = LatentDirichletAllocation(n_components=n_components, max_iter = max_iter)
    lda_model.fit(vectorized_texts)

    return lda_model


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
