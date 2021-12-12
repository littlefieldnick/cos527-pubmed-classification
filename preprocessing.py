#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:00:36 2021

@author: nicklittlefield

=========================================================

preprocessing.py: Contains the text preprocessing code to preprocess the abstracts 
                  downloaded from Pubmed.  

"""

import numpy as np
import pandas as pd

import string
import nltk

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split

def preprocess_document(text_data):
    # Helpers for preprocessing
    stop_list = stopwords.words('english')
    tokenizer = RegexpTokenizer("\w+")
    lem = WordNetLemmatizer() 
    stem = PorterStemmer() 
    
    
    # Perform preprocessing
    # 1. Convert to lowercase
    # 2. Split into tokens to preprocess individual words
    # 3. Lemmatization
    # 4. Stemming
    # 5. Stop words and numerical numbers
    
    doc = text_data.lower()
    tokens = tokenizer.tokenize(doc)
    tokens = [lem.lemmatize(tok) for tok in tokens]
    tokens = [stem.stem(tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in stop_list and not tok.isnumeric()]
        
    # 6. Join and return tokens back into string to pass to CountVectorizer and form bag of words
    return " ".join(tokens)


def gen_train_test_split(abstracts, test_size=0.2):
    X = abstracts["text"]
    y = abstracts["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.apply(preprocess_document)
    X_test = X_test.apply(preprocess_document)

    train = pd.DataFrame({
        "text": X_train,
        "class": y_train
    })

    test = pd.DataFrame({
        "text": X_test,
        "class": y_test    
    })

    return train, test





