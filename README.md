# Pubmed Abstract Classification
## Nick Littlefield

### Overview
New biomedical research articles are being published every day. By using text mining on large scale scientific literature, we can discover new knowledge to better understand human diseases and help to improve the quality of disease diagnosis, prevention, and treatment. In this project, the goal is to build a text classification system that downloads abstracts from Pubmed to extract and classify four different diseases: lyme disease, acute rheumatic arthritis, knee osteoarthritis, and cardiovascular abnormalities. 

### Materials and Methods
There are three tiers to to this classification system:
1. Data Access
2. Text Preprocessing
3. Machine Learning and Classification

#### Tier 1: Data Access
Roughly 77,000 abstracts were downloaded from Pubmed using the `Biopython` library. These abstracts along with their corresponding class were saved to a CSV file for later preprocessing. Any abstracts downloaded that had no text were discarded. 

#### Tier 2: Text Preprocessing
Text processing was done on the saved abstracts using the `nltk` library. The following preprocessing steps were done:
1. Removal of punctuation and numbers
2. Text normalization: all text converted to lowercase
3. Tokenization
4. Lemmatization and stemming
5. Preprocessed text rejoined into a string for use in the machine learning tier

After preprocessing the abstracts, they were split into a training and test set using a 80/20 split. The train and test set were saved in individual CSV files for use in the machine learning tier. The training set had ~62,000 abstracts while the training set had ~15,000 abstracts.

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
The classification report on the test set for Naive Bayes is shown below:

```
                               precision    recall  f1-score   support

abnormalities, cardiovascular       0.98      0.97      0.98     11370
    acute rheumatic arthritis       0.37      0.91      0.53       369
                disease, lyme       1.00      0.59      0.74       661
          knee osteoarthritis       0.99      0.94      0.97      3149

                     accuracy                           0.95     15549
                    macro avg       0.84      0.85      0.80     15549
                 weighted avg       0.97      0.95      0.96     15549
```

The confusion matrix for the test set is:

```
                               disease, lyme  abnormalities, cardiovascular  knee osteoarthritis  acute rheumatic arthritis
disease, lyme                            389                             81                    1                        190
abnormalities, cardiovascular              0                          11069                   10                        291
knee osteoarthritis                        0                             84                 2970                         95
acute rheumatic arthritis                  0                             27                    5                        337
```

#### Logistic Regression
The classification report on the test set for logistic regression is shown below:

```
                               precision    recall  f1-score   support

abnormalities, cardiovascular       0.99      0.99      0.99     11370
    acute rheumatic arthritis       0.78      0.89      0.83       369
                disease, lyme       1.00      0.94      0.97       661
          knee osteoarthritis       0.99      0.98      0.99      3149

                     accuracy                           0.99     15549
                    macro avg       0.94      0.95      0.94     15549
                 weighted avg       0.99      0.99      0.99     15549
```

The confusion matrix for the test set is:

```
                               disease, lyme  abnormalities, cardiovascular  knee osteoarthritis  acute rheumatic arthritis
disease, lyme                            621                             29                    5                          6
abnormalities, cardiovascular              1                          11294                    9                         66
knee osteoarthritis                        0                             55                 3073                         21
acute rheumatic arthritis                  0                             39                    2                        328
```

#### Support Vector Machine
The classification report on the test set for the support vector machine is shown below:

```
                               precision    recall  f1-score   support

abnormalities, cardiovascular       0.99      0.99      0.99     11370
    acute rheumatic arthritis       0.86      0.88      0.87       369
                disease, lyme       1.00      0.97      0.98       661
          knee osteoarthritis       0.99      0.99      0.99      3149

                     accuracy                           0.99     15549
                    macro avg       0.96      0.96      0.96     15549
                 weighted avg       0.99      0.99      0.99     15549

```

The confusion matrix for the test set is:

```
                               disease, lyme  abnormalities, cardiovascular  knee osteoarthritis  acute rheumatic arthritis
disease, lyme                            640                             15                    4                          2
abnormalities, cardiovascular              0                          11312                   16                         42
knee osteoarthritis                        2                             34                 3103                         10
acute rheumatic arthritis                  0                             41                    4                        324
```

### Conclusions
Out of the three models, the SVM and logistic regression model had the highest accuracy and performance for predicting all four of the classes, with both models having 99% accuracy. Naive Bayes performed the worst out of the three models, with 95% accuracy. When it comes to how well the model does with precision and recall, the SVM model has the best performance.  

For the Naive Bayes model, it had the hardest time classifying the acute rheumatic arthritis class. It correctly predicts this class 37% of the time and correctly identified 91% of actual acute rheumatic arthritis abstracts correctly. As the different models are tried this goes up. For logistic regression, the model correctly predicts the class 77% of the time and correctly identifies 87% all the acute rheumatic arthritis abstracts. The SVM model substantially increases this. It correctly predicts the class 86% of the time and correctly identifies 88% of all the acute rheumatic arthritis abstracts. 

The Naive Bayes model also struggled with the lyme disease class. This model correctly identified lyme disease 100% of the time but only correctly identified 51% of the actual lyme disease abstracts. Logistic regression and SVM both did well with this class, however, again the SVM model had a better time predicting it. The SVM predicts lyme disease 99% of the time and correctly identified 99% of all of the lyme disease abstracts.  None of the models had trouble with the cardiovascular abnormalities and knee osteoarthritis classes. 

Out of the three models, it can therefore be concluded that if this classification system were to be used in a real world application, the SVM model would be the best model to represent these set of abstracts. 

### Discussion
The hardest part to implement for this project was the data extraction. Two methods were tried: beautifulsoup and Biopython. Biopython was the easiest and fastest method to use, and was the method that was selected. When using beautifulsoup, the amount of time to extract the abstracts was a couple of hours. Along with this because of the setup of Pubmed, only the first 50,000 abstracts could be downloaded. The changing of pages also needed to be handled so more information needed to be extracted than just the abstracts. HTML within the abstracts were also difficult to handle. Biopython was the fastest as it uses a search mechanism called entrez which fetches the results of a query and returns them in an XML format that can be parsed. Compared to beautifulsoup this method takes minutes (~10 minutes) compared to the hours it took to download the information using beautifulsoup. 

When looking at the three models, we know SVM had the best performance and this can be understood when thinking about how the models are working. Naive Bayes is a probabalistic model. It assumes that the features are independent, which isn't the case in this problem (and isn't most of the time) and depends only on prior knowledge to make predictions.  Logistic regression is a linear model and the decision boundaries the model makes are all linear which is a disadvantage if the features for the model are not linearly separable. The advantages come from the support vector machine. This model has three different ways of modeling the data by using different kernels (the kernel trick). These kernel options are: linear, polynomial, and the radial basis function. These kernels try to find the best margin (or line) that separates the data which in return reduces the amount of error when making classifications. Unlike logistic regression though, these margins do not need to be linear and can be controlled using smoothing parameters and different degrees (for polynomial kernels only). Therefore, when it comes to this problem, it makes sense that the SVM would work well since the data is most likely not linearly separable. 

One other method that was tried (and at least worth mentioning) was using word2vec instead of bag of words. This method, however, wasn't successful, as the results were very poor and not worth reporting (a single Naive Bayes model has an accuracy of 29%). When looking at the vectors generated by using a pretrained word2vec model (glove) this actually made sense. Most of the vectors in the word2vec representation for tokens in an abstract were zeroes. This made sense because a lot of the words whose vectors were zeroes were medical terms not contained in the pretrained word2vec model. Since these terms are vital for the model to distinguish between different diseases the model therefore couldn't make good predictions. 


### Future Outlooks
Going forward there are a couple of things that can be done to improve the models. The first thing that can be done, is a grid search to find the correct parameters to maximize the accuracy of the predictions. For this project, the default settings were used and nothing was done to fine tune the models. Along with this, would be to continue trying to improve the word2vec model performances. One method for this to be done, is to either use pretrained embeddings on medical text, or to train the word2vec model from scratch on the Pubmed corpus and extract those vectors as features for the different models. 

Next, the models are limited to only the abstracts downloaded when first run. Therefore, a method to download additional abstracts, such as on a daily, biweekly, or monthly occurance, could be added. Then the model can be retrained using the previously downloaded abstracts and the newly downloaded ones. This would increase the sample size being used to train the model and improve the performance. 

Lastly, the application is restricted to using four different classes of diseases. It would be useful to have a way to expand the classes of abstracts being downloaded. Additionally, other machine learning models such as decision trees or even deep learning could be used to try and train additional models that may perform better. 

