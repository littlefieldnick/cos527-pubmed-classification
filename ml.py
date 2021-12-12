#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:41:54 2021

@author: nicklittlefield

=========================================================
ml.py: Machine learning algorithms for abstract classification and helper functions
       for loading/saving models and evaluating models. 

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
    """
    Saves trained models to specifed location
    
    Parameters
    ----------
    model_pth : 
        Path to save the model
    nb_pipeline :
        Trained Naive Bayes pipeline
    log_pipeline :
        Trained logistic regression pipeline
    svm_pipeline : 
        Trained support vector machine pipeline

    Returns
    -------
    None.

    """
    # Save each of the models to be loaded later one
    pickle.dump(nb_pipeline, open(model_pth + "naive_bayes.sav", "wb"))
    pickle.dump(log_pipeline, open(model_pth + "logistic.sav", "wb"))
    pickle.dump(svm_pipeline, open(model_pth + "svm.sav", "wb"))


def load_models(model_pth):
    """
    Loads models from specified path

    Parameters
    ----------
    model_pth : 
        Path to models to load

    Returns
    -------
    nb_pipeline : 
        Loaded Naive Bayes pipeline
    log_pipeline : 
        Loaded logistic regression pipeline
    svm_pipeline : 
        Loaded SVM pipeline

    """
    # Load all models
    nb_pipeline = pickle.load(open(model_pth + "naive_bayes.sav", "rb"))
    log_pipeline = pickle.load(open(model_pth + "logistic.sav", "rb"))
    svm_pipeline = pickle.load(open(model_pth + "svm.sav", "rb"))
    
    # Return loaded models
    return nb_pipeline, log_pipeline, svm_pipeline

def print_confusion_matrix(y_true, y_pred, labels):
    """
    Prints the confusion matrix for a model

    Parameters
    ----------
    y_true : 
        True class labels
    y_pred : 
        Predicted labels
    labels : 
        Class label names

    Returns
    -------
    None.

    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Convert to dataframe for formated display
    df_form = pd.DataFrame(cm, columns=labels, index=labels)
    print("Confusion Matrix:")
    print(df_form)
    
    
def print_classification_report(y_true, y_pred, model_name, dataset="Train"):
    """
    Prints the classification report for a model

    Parameters
    ----------
    y_true : 
        True class labels
    y_pred : TYPE
        Predicted labels
    model_name : TYPE
        Name of the model results are from
    dataset : optional
        Dataset being used. The default is "Train".

    Returns
    -------
    None.

    """
    print(model_name, "--", dataset, "Results:")
    
    # Print classification report 
    print(classification_report(y_true, y_pred))

def eval_model(model, test_set, labels, model_name):
    """
    Evaluates a model

    Parameters
    ----------
    model : 
        Pipeline for a given model
    test_set : 
        Test dataset
    labels : TYPE
        Label names for confusion matrix
    model_name : 
        Name of trained model

    Returns
    -------
    None.

    """
    
    # Get predictions
    y_pred = model.predict(test_set["text"])
    
    # Extract true class
    y_true = test_set["class"]
    
    # Display results
    print_classification_report(y_true, y_pred, model_name, dataset="Test")
    print_confusion_matrix(y_true, y_pred, labels)

    
def train_nb_pipeline(train_data):
    """
    Trains a naive bayes model.

    Parameters
    ----------
    train_data : 
        Training dataframe

    Returns
    -------
    nb_pipeline : 
        Trained naive bayes model
    """
    
    # Pipeline
    nb_pipeline = Pipeline([('bow', CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("sampler", RandomOverSampler(sampling_strategy='minority', random_state=42)),
                            ("naive_bayes", MultinomialNB())])


    # Train the model
    nb_model = nb_pipeline.fit(train_data["text"], train_data["class"])
    
    # Get train predictions
    pred = nb_model.predict(train_data["text"])

    # Display results for training
    print_classification_report(train_data["class"], pred, "Naive Bayes")

    # Return trained pipeline
    return nb_pipeline

def train_logistic_regression_pipeline(train_data):
    """
    Train a logistic regression model

    Parameters
    ----------
    train_data : 
        Training dataframe

    Returns
    -------
    logreg_pipeline : 
        Trained logistic regression model

    """
    # Pipeline
    logreg_pipeline = Pipeline([('bow', CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("sampler", RandomOverSampler(sampling_strategy='minority', random_state=42)),
                            ("logistic", LogisticRegression(max_iter=1000))])


    # Train the model
    log_model = logreg_pipeline.fit(train_data["text"], train_data["class"])
    
    # Get train predictions
    pred = log_model.predict(train_data["text"])

    # Display the results for training
    print_classification_report(train_data["class"], pred, "Logistic Regression")
    
    # Return trained pipeline
    return logreg_pipeline

def train_svm_pipeline(train_data):
    """
    Train a support vector machine model

    Parameters
    ----------
    train_data : 
        Training dataframe

    Returns
    -------
    svm_pipeline : 
        Trained logistic regression model
    """
    # Pipeline
    svm_pipeline = Pipeline([('bow', CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("sampler", RandomOverSampler(sampling_strategy='minority', random_state=42)),
                            ("svm", LinearSVC())])

    # Train the model
    svm_model = svm_pipeline.fit(train_data["text"], train_data["class"])
    
    # Get train predictions
    pred = svm_model.predict(train_data["text"])

    # Display the results
    print_classification_report(train_data["class"], pred, "Support Vector Machine")
    
    # Return trained pipeline
    return svm_pipeline


