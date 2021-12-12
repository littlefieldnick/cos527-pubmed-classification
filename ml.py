#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:41:54 2021

@author: nicklittlefield
"""

import numpy as np
import pandas as pd
import pickle

from imblearn.pipeline import Pipeline 
from imblearn.over_sampling  import RandomOverSampler


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def save_models(model_pth, nb_pipeline, log_pipeline, svm_pipeline):
    pickle.dump(nb_pipeline, open(model_pth + "naive_bayes.sav", "wb"))
    pickle.dump(log_pipeline, open(model_pth + "logistic.sav", "wb"))
    pickle.dump(svm_pipeline, open(model_pth + "svm.sav", "wb"))


def load_models(model_pth):
    nb_pipeline = pickle.load(open(model_pth + "naive_bayes.sav", "rb"))
    log_pipeline = pickle.load(open(model_pth + "logistic.sav", "rb"))
    svm_pipeline = pickle.load(open(model_pth + "svm.sav", "rb"))
    
    return nb_pipeline, log_pipeline, svm_pipeline

def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_form = pd.DataFrame(cm, columns=labels, index=labels)
    print("Confusion Matrix:")
    print(df_form)
    
    
def print_classification_report(y_true, y_pred, model_name, dataset="Train"):
    print(model_name, "--", dataset, "Results:")
    print(classification_report(y_true, y_pred))

def eval_model(model, test_set, labels, model_name):
    y_pred = model.predict(test_set["text"])
    y_true = test_set["class"]
    
    print_classification_report(y_true, y_pred, model_name, dataset="Test")
    print_confusion_matrix(y_true, y_pred, labels)

    
def train_nb_pipeline(train_data):
    nb_pipeline = Pipeline([('bow', CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("sampler", RandomOverSampler(sampling_strategy='minority', random_state=42)),
                            ("naive_bayes", MultinomialNB())])



    nb_model = nb_pipeline.fit(train_data["text"], train_data["class"])
    pred = nb_model.predict(train_data["text"])

    print_classification_report(train_data["class"], pred, "Naive Bayes")

    return nb_pipeline

def train_logistic_regression_pipeline(train_data):
    logreg_pipeline = Pipeline([('bow', CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("sampler", RandomOverSampler(sampling_strategy='minority', random_state=42)),
                            ("logistic", LogisticRegression(max_iter=1000))])



    log_model = logreg_pipeline.fit(train_data["text"], train_data["class"])
    pred = log_model.predict(train_data["text"])

    print_classification_report(train_data["class"], pred, "Logistic Regression")
    
    return logreg_pipeline

def train_svm_pipeline(train_data):
    svm_pipeline = Pipeline([('bow', CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("sampler", RandomOverSampler(sampling_strategy='minority', random_state=42)),
                            ("svm", LinearSVC())])


    svm_model = svm_pipeline.fit(train_data["text"], train_data["class"])
    pred = svm_model.predict(train_data["text"])

    print_classification_report(train_data["class"], pred, "Support Vector Machine")
    
    return svm_pipeline


