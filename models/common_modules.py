import pandas as pd
import re
from collections import defaultdict
import spacy
import nltk
from nltk.util import ngrams
import numpy as np



def csv_to_df(csvpath,
              sample_ratio=1.0,
              lang='EN',
              random_seed=3195
              ):
    """
    Convert a csv to pre-processed sample DataFrame
    Args:
        csvpath: path of the csv file
        sample_ratio: 0.1 for 10%, for example. 100% by default.
        lang: 'EN' or 'FR'
        random_seed: int used in sampling to make it reproducible.
    Returns:
        a pandas DataFrame
    """

    def text_to_word_list(text):
        """
        helper function to pre-process and convert texts to a list of words.
        Adapted from Jan's code which refers work of Rafał Wójcik:
        towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
        """

        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()
        return text

    full_df = pd.read_csv(csvpath)
    single_lang_df = full_df[full_df["Lang"] == lang]

    # Only keep Comments and Tags
    single_lang_df = single_lang_df[["Comment", "Tags"]]
    cleaned_df = single_lang_df.copy()
    cleaned_df.Comment = single_lang_df.Comment.apply(
        lambda x: text_to_word_list(x)
    )
    cleaned_df.Comment = cleaned_df.Comment.str.join(' ')

    sample_df = cleaned_df.sample(
        random_state=random_seed,
        frac=sample_ratio
    )

    return sample_df


def replace_unicode_quote(input_str):
    """
    Replace Unicode quote symbol into ASCII equivalent in input string
    Args:
        input_str(str): input string
    Returns:
        a string with ASCII quote symbol
    """
    return input_str.replace('‘', "'").replace('’', "'")
