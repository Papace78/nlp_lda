#Preprocessing
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = set(list(stopwords.words('english')))


def clean_data(x: str) -> str:

    x = "".join([w for w in x if not w.isdigit()])

    for punctuation in string.punctuation:
        x = x.replace(punctuation,'')

    x = x.lower()
    x = x.strip()
    return x

def preprocess(x : str) -> str:

    x = word_tokenize(x)

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in x
    ]

    x = " ".join(w for w in lemmatized if not w in stop_words)

    return x
