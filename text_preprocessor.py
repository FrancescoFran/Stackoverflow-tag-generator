import numpy as np
import pandas as pd
import re
import contractions
import spacy
import unidecode
import nltk
from langdetect import detect, DetectorFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')


def lang_check(x):
    """Function to check language
        Parameters:
        x:string
    """

    lang = detect(x)
    if lang != 'en':
        return 'Not english'
    else:
        return x


def cleaner(x):
    """Function to clean text
    Parameters:
        x:string
    """

    # eliminate urls
    x = re.sub(r'http\S+', '', x)
    # eliminate HTML
    soup = BeautifulSoup(x, "html.parser")
    x = soup.get_text()
    # eliminate english contractions
    x = contractions.fix(x)
    # POS tagging
    nlp = spacy.load('en_core_web_sm', exclude=['ner', 'parser', 'lemmatizer'])
    pos_list = ["NOUN", "PROPN"]
    doc = nlp(x)
    list_text = []
    for token in doc:
        if token.pos_ in pos_list:
            list_text.append(token.text)
    x = " ".join(list_text)
    # eliminate not alphanumeric
    x = re.sub('[^a-z+#A-Z]', ' ', x)
    # eliminate accented characters
    x = unidecode.unidecode(x)
    # conversion to lower characters
    x = x.lower()
    # tokenize text
    x = word_tokenize(x)
    # eliminate stopwords
    sw_nltk = stopwords.words('english')
    x = [word for word in x if word not in sw_nltk]
    # lemmatization
    lemma = WordNetLemmatizer()
    x = [lemma.lemmatize(word=w, pos='v') for w in x]
    # eliminate tokens longer than 2 except 'c'
    x = [i for i in x if (len(i) > 2 or i == 'c')]
    return x

