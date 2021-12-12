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

# Email is needed to use the entrez
entrez.email = "nickolas.littlefield@maine.edu"

class PubmedEntrezAPI:
    def __init__(self, terms, entrez_params, date_start, date_end=None, batch_size=100, track_counts=True):
        """
        Parameters
        ----------
        terms : 
            list of MeSH Terms to search for
        entrez_params : 
            Parameters for entrez search
        date_start : 
            Starting publication date for finding abstracts
        date_end : optional
            Ending publication date for finding abstracts. The default is None.
        batch_size :  optional
            Download batch size The default is 100.
        track_counts : optional
            Keep track of the number of abstracts for each term. The default is True.

        Returns
        -------
        None.

        """
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
        
        # Query formatter for making queries
        self.query_format = "{text}[MESH] AND {start_date}[Date - Publication] : {end_date}[Date - Publication]"
        
    def load_data(self, verbose=True):
        """
        Loads abstracts from pubmed

        Parameters
        ----------
        verbose :  optional
            Print download process. The default is True.

        Returns
        -------
        None.

        """
        results = []
        labels = []
        
        # Loop through terms to download
        for term in self.terms:
            # Form the query
            query_term = self.query_format.format(text=term, start_date=self.start_date, 
                                      end_date=self.end_date)
            
            # Get the results of the query
            search_results = entrez.read(entrez.esearch("pubmed", query_term,
                                                        **self.entrez_params))
            # Extract the count
            count = int(search_results["Count"])
            out_data = []
            
            # Download abstracts in batches of batch_size
            for start in range(0, count, self.batch_size):
                end = min(count, start + self.batch_size)
                if verbose:
                    print(term, "-- Downloading records %i to %i" % (start + 1, end))
                
                # Fetch the results using the previous history stored in search_results
                fetch_handle = entrez.efetch("pubmed", 
                                             retstart=start,
                                             retmax=self.batch_size,
                                             webenv=search_results["WebEnv"],
                                             query_key=search_results["QueryKey"],
                                             **self.entrez_params)
                
                # Read the results, extract the abstract and store in results
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
        
        # Make dataframe to store abstact/classes
        self.df = pd.DataFrame(data={"text": results,
                                     "class": labels
                               })

    def get_data(self):
        """
        Get the downloaded abstract dataset

        Returns
        -------
        Downloaded abstract dataset

        """
        # return the downloads and associated class label
        return self.df





