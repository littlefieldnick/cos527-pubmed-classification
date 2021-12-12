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

train_models = False
data_pth = "/Users/nicklittlefield/Desktop/cos527-pubmed-classification/data/"
model_pth = "/Users/nicklittlefield/Desktop/cos527-pubmed-classification/saved_models/"

def data_access_tier(terms, entrez_params, start_date, end_date=None, batch_size=100):
    # Create Pubmed API object to download abstracts
    api = PubmedEntrezAPI(terms, entrez_params, date_start=start_date, date_end=end_date, batch_size=batch_size)

    # Download abstracts
    api.load_data()
    
    # Get the abstracts
    abstracts = api.get_data()
    
    # Save them to file
    abstracts.to_csv(data_pth + "abstract.csv", index=False)
    

def preprocess_tier(abstract_pth):
    abstracts = pd.read_csv(abstract_pth)
    abstracts.dropna(inplace=True) # NaNs are in the dataset... Not sure why, so lets drop them.
    train, test = gen_train_test_split(abstracts)
    
    # Save the outputs
    train.to_csv(data_pth + "train.csv", index=False)
    test.to_csv(data_pth + "test.csv", index=False)

    return train, test

def ml_train_tier(train_data, save_pipelines=True):
    nb_pipeline = train_nb_pipeline(train)
    log_pipeline = train_logistic_regression_pipeline(train)
    svm_pipeline = train_svm_pipeline(train)
    
    if save_pipelines:
        save_models(model_pth, nb_pipeline, log_pipeline, svm_pipeline)

    return nb_pipeline, log_pipeline, svm_pipeline

def ml_eval_tier(test_data, nb, log, svm, labels):
    eval_model(nb, test_data, labels, "Naive Bayes")
    eval_model(log, test_data, labels, "Logistic Regression")
    eval_model(svm, test_data, labels, "Support Vector Machine")


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

train.dropna(inplace=True)
test.dropna(inplace=True)

if train_models:
    nb_pipeline, log_pipeline, svm_pipeline = ml_train_tier(train)
else:
    nb_pipeline, log_pipeline, svm_pipeline = load_models(train)

ml_eval_tier(test, nb_pipeline, log_pipeline, svm_pipeline, terms)

