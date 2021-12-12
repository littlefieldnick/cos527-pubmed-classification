#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:01:12 2021

@author: nicklittlefield

=========================================================
PubmedEntrezAPI: Class that downloads abstracts from Pubmed using the Biopython api
                 Serves as a wrapper to hold all downloaded data in a dataframe. 
"""


from Bio import Entrez as entrez
import pandas as pd

entrez.email = "nickolas.littlefield@maine.edu"

class PubmedEntrezAPI:
    def __init__(self, terms, entrez_params, date_start, date_end=None, batch_size=100, track_counts=True):
        
        self.terms = terms
        self.entrez_params = entrez_params

        self.start_date = date_start
        
        if date_end is None:
            self.end_date = "3000" # 3000 is represents present date
        else:
            self.end_date = date_end
            
        self.batch_size = batch_size
        self.df = None
        
        self.track_counts=track_counts
        
        if track_counts:
            self.counts=[]
        
        self.query_format = "{text}[MESH] AND {start_date}[Date - Publication] : {end_date}[Date - Publication]"
        
    def load_data(self, max_cutoff=None, verbose=True):
        results = []
        labels = []
        
        # Loop through terms to download
        for term in self.terms:
            query_term = self.query_format.format(text=term, start_date=self.start_date, 
                                      end_date=self.end_date)
            
            search_results = entrez.read(entrez.esearch("pubmed", query_term,
                                                        **self.entrez_params))

            count = int(search_results["Count"])
            out_data = []
            
            # Download abstracts in batches of batch_size
            for start in range(0, count, self.batch_size):
                end = min(count, start + self.batch_size)
                if verbose:
                    print(term, "-- Downloading records %i to %i" % (start + 1, end))
                fetch_handle = entrez.efetch("pubmed", 
                                             retstart=start,
                                             retmax=self.batch_size,
                                             webenv=search_results["WebEnv"],
                                             query_key=search_results["QueryKey"],
                                             **self.entrez_params)
                data = entrez.read(fetch_handle)
                fetch_handle.close()
                out_data.extend([res["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
                                for res in data["PubmedArticle"] if "Abstract" in res["MedlineCitation"]["Article"]])
            
            results.extend(out_data)
            labels.extend([term] * len(out_data))
            
            # Keep track of counts
            if self.track_counts:
                self.counts.append(len(out_data))
                
            if verbose:
                print(term, "-- Extraction complete! Downloaded ", len(out_data), "abstracts.")
        
        # Make dataframe to store
        
      
        self.df = pd.DataFrame(data={"text": results,
                                     "class": labels
                               })

    def get_data(self):
        # returns the downloads and associated class label
        return self.df





