#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:00:28 2021

@author: nicklittlefield

=========================================================

main.py: Main script that runs abstract downloads and runs machine learning classification
         algorithms. 
"""

import os
import pickle

import numpy as np
import pandas as pd

from download_abstract import PubmedEntrezAPI
from preprocessing import gen_train_test_split
from ml import save_models, load_models, train_nb_pipeline, train_svm_pipeline, \
        train_logistic_regression_pipeline, \
        eval_model

# Train models: False if models are already trained. Set to True to train models
train_models = False

# Paths to save abstracts/train/test data and models
data_pth = "/Users/nicklittlefield/Desktop/cos527-pubmed-classification/data/"
model_pth = "/Users/nicklittlefield/Desktop/cos527-pubmed-classification/saved_models/"

def data_access_tier(terms, entrez_params, start_date, end_date=None, batch_size=100):
    """
    Tier 1 -- Data Access

    Parameters
    ----------
    terms : 
        List of MeSH Terms to search for
    entrez_params : 
        Dictionary of parameters to use for entrez. 
    start_date : 
        Starting publication date for search
    end_date :  optional
        DEnding publication date for search The default is None.
    batch_size :  optional
        Download The default is 100.

    Returns
    -------
    abstracts:
        Abstract dataset

    """
    # Create Pubmed API object to download abstracts
    api = PubmedEntrezAPI(terms, entrez_params, date_start=start_date, date_end=end_date, batch_size=batch_size)

    # Download abstracts
    api.load_data()
    
    # Get the abstracts
    abstracts = api.get_data()
    
    # Save them to file
    abstracts.to_csv(data_pth + "abstract.csv", index=False)
    
    # Return abstract dataframe
    return abstracts 

def preprocess_tier(abstract_pth):
    """
    Tier 2 -- Preprocessing. Process abstracts downloaded and save train/test splits

    Parameters
    ----------
    abstract_pth : 
        Path to saved abstracts

    Returns
    -------
    train : TYPE
        Train dataset
    test : TYPE
        Test dataset

    """
    
    abstracts = pd.read_csv(abstract_pth)
    abstracts.dropna(inplace=True) # NaNs are in the dataset... Not sure why, so lets drop them.
    train, test = gen_train_test_split(abstracts)
    
    # Save the outputs
    train.to_csv(data_pth + "train.csv", index=False)
    test.to_csv(data_pth + "test.csv", index=False)

    return train, test

def ml_train_tier(train_data, save_pipelines=True):
    """
    Tier 3 -- Machine learning. Fits Naive Bayes, Logistic Regression, and SVM

    Parameters
    ----------
    train_data : 
        Training dataset
    save_pipelines : optional
        Save models that have been trained. The default is True.

    Returns
    -------
    nb_pipeline : 
        Naive Bayes pipeline.
    log_pipeline : 
        Logistic Regression pipeline
    svm_pipeline : 
        SVM pipeline
    """
    
    # Train pipelines
    nb_pipeline = train_nb_pipeline(train)
    log_pipeline = train_logistic_regression_pipeline(train)
    svm_pipeline = train_svm_pipeline(train)
    
    # Save them
    if save_pipelines:
        save_models(model_pth, nb_pipeline, log_pipeline, svm_pipeline)

    # Return trained pipelines
    return nb_pipeline, log_pipeline, svm_pipeline

def ml_eval_tier(test_data, nb, log, svm, labels):
    """
    Evaluate machine learning models

    Parameters
    ----------
    test_data : TYPE
        DESCRIPTION.
    nb : TYPE
        DESCRIPTION.
    log : TYPE
        DESCRIPTION.
    svm : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    eval_model(nb, test_data, labels, "Naive Bayes")
    eval_model(log, test_data, labels, "Logistic Regression")
    eval_model(svm, test_data, labels, "Support Vector Machine")


# Make sure data and model directories exist
if not os.path.exists(data_pth):
    os.mkdir(data_pth)
    
if not os.path.exists(model_pth):
    os.mkdir(model_pth)
    
    
#############################
# Tier 1: Data Access       #
#############################
    
# Parameters for entrez search
entrez_params = {
    "retmode": "xml",
    "rettype": "abstract",
    "usehistory": "y",
}

# MeSH Terms for Pubmed
terms = ["disease, lyme", "abnormalities, cardiovascular", "knee osteoarthritis",
         "acute rheumatic arthritis"]


# Check if abstracts are downloaded, if not download them
if not os.path.exists(data_pth + "abstract.csv"):
    abstracts = data_access_tier(terms, entrez_params, "01/01/2010", batch_size=5000)
    

#############################
# Tier 2: Preprocessing     #
#############################

# Check if train and test splits if they don't exist generate the splits and save them
if not os.path.exists(data_pth + "train.csv") or not os.path.exists(data_pth + "test.csv"):
    train, test = preprocess_tier(data_pth + "abstract.csv")
else:
    train = pd.read_csv(data_pth + "train.csv")
    test = pd.read_csv(data_pth + "test.csv")


#############################
# Tier 3: Machine Learning  #
#############################

# Drop any nas that may not have passed the drop filter
train.dropna(inplace=True)
test.dropna(inplace=True)

# Train models if train_models is True, else load trained models
if train_models:
    nb_pipeline, log_pipeline, svm_pipeline = ml_train_tier(train)
else:
    nb_pipeline, log_pipeline, svm_pipeline = load_models(model_pth)

# Evaluate the models
ml_eval_tier(test, nb_pipeline, log_pipeline, svm_pipeline, terms)

