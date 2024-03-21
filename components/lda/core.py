import pandas as pd

from reader import get_texts, concatenate_text
from preprocessor import clean_data, preprocess
from lda import tfid_vectorizer, lda, print_topics


def main(ngram_range: tuple = (1, 1), n_components: int = 2, max_iter: int = 100):

    texts = get_texts()
    concat_texts = concatenate_text(texts)

    cleaned_texts = concat_texts.map(clean_data)
    preproc_texts = cleaned_texts.map(preprocess)

    vectorizer, vectorized_texts = tfid_vectorizer(
        preproc_texts[0], ngram_range=ngram_range
    )

    lda_model = lda(vectorized_texts, n_components=n_components, max_iter=max_iter)

    document_topic_mixture = lda_model.transform(vectorized_texts)

    """
    document_topic_dataframe = pd.DataFrame(data=document_topic_mixture, columns = ["topic_0", "topic_1"])
    print(f'Document topic mixture sample: \n{document_topic_dataframe.sample(5)}')


    topic_word_mixture = pd.DataFrame(
        lda_model.components_, columns=vectorizer.get_feature_names_out()
    )
    print(f'Topic word mixture sample: \n{topic_word_mixture}')
    """

    print_topics(lda_model, vectorizer, 5)
