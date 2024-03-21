import os
import pandas as pd

texts_path = os.path.join('textes')

def get_texts() -> dict:
    texts = {}
    for text in os.listdir(texts_path):
        texts[text] = pd.read_csv(os.path.join(texts_path, text), delimiter = "/n", engine = 'python', header = None)
    return texts

def concatenate_text(texts: dict) -> pd.DataFrame:
    return pd.concat(texts.values())
