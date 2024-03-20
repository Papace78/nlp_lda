#Preprocessing
import string

from nltk.corpus import stopwords
from nltk import RegexpTokenizer

import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop


nlp_fr = spacy.load("fr_core_news_md")
tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
stop_words = set(list(fr_stop)+list(stopwords.words('french')))


def clean_data(x: str) -> str:
    x = "".join([w for w in x if not w.isdigit()])
    for punctuation in string.punctuation:
        if punctuation == "'":
            continue
        x = x.replace(punctuation,'')

    x = x.lower()
    x = x.strip()
    return x

def preprocess(x : str) -> str:

    x = tokenizer.tokenize(x)

    x = " ".join([w for w in x if not w in stop_words and len(w)>1])

    tokens = nlp_fr(x)
    x = " ".join([token.lemma_ for token in tokens])

    return x
