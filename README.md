# Pubmed Abstract Classification
## Nick Littlefield

### Overview
New biomedical research articles are being published every day. By using text mining on large scale scientific literature, we can discover new knowledge to better understand human diseases and help to improve the quality of disease diagnosis, prevention, and treatment. In this project, the goal is to build a text classification system that downloads abstracts from Pubmed to extract and classify four different diseases: lyme disease, acute rheumatic arthritis, and cardiovascular abnormalities. 

### Materials and Methods
There are three tiers to to this classification system:
1. Data Access
2. Text Preprocessing
3. Machine Learning and Classification

#### Tier 1: Data Access
Roughly 77,000 abstracts were downloaded from Pubmed using the Biopython library. These abstracts along with their corresponding class were saved to a CSV file for later preprocessing. Any abstracts that downloaded that had no text were discarded. 

#### Tier 2: Text Preprocessing
Text processing was done on the saved abstracts using the `nltk` library. The following preprocessing steps were done:
1. Removal of punctuation and numbers
2. Text normalization: all text converted to lowercase
3. Text was split into lists of tokens
4. Lemmatization and stemming applied to tokens
5. Preprocessed text rejoined into a string for use in the machine learning tier

After preprocessing the abstracts, they were split into a training and test set using a 80/20 split. The train and test set were saved in individual CSV files for use in the machine learning tier.

#### Tier 3: Machine Learning
Three machine learning algorithms were used for classification via the `sklearn` library. These algorithms were:
1. Naive Bayes
2. Logistic Regression
3. Support Vector Machines

The abstracts used contains a large imbalance between the different classes. To handle this imbalance the 'imblearn' library was used to oversample the minority classes. 

To perform the training process a pipeline was built to perform a series of steps. These steps were:
1. Convert training data to a bag of words representation
2. Apply TF-IDF to the bag of words representation
3. Oversample the minority classes
4. Train the machine learning algorithm.

By using this pipeline when it comes time to evaluate the model, the pipeline will also convert the test abstracts to a Bag of Words representation and apply TF-IDF. The oversampling step is ignored when predictions are made. 

### Results
The results for each of the machine learning algorithms are below. For each model, the confusion matrix, F1 score, precision, recall, and support are provided. 

#### Naive Bayes

#### Logistic Regression

#### Support Vector Machine

### Conclusions

### Discussion

### Future Outlooks
