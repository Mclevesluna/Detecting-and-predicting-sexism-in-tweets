READ ME - SEXISM PROJECT
MANUELA CLEVES


1. Project Overview and relevant insights/conclusions
Our project consists of a series of algorithms to analyze sexist bias in tweets in order to 1) predict it and 2) understand the relationships between words commonly found together in sexist tweets.

To accomplish this, we used data sets prepared by Francisco Rodriguez-Sanchez, Jorge Carrillo-de-Albornoz, Laura Plaza1, Julio
Gonzalo, Paolo Rosso, Trinidad Donoso and Miriam Comet for their EXIST 2021 SEXism Identification in Social neTworks Task Guidelines.

Our training and test data sets contain the following:

- "Test_case": tag needed in the EvALL framework for evaluating classification tasks. In EXIST 2021, this tag is set to “EXIST2021”.
- "Id": denotes a unique identifier of the tweet.
- "Source": defines the social network where the text was crawled, “twitter” and "gab" in the training set.
- "Language": denotes the languages of the text (“en” or “es”).
- "Text": represents the text of the tweet.
- "Task1": indicates if the tweet is sexist (“sexist”) or not (“non-sexist”).
- "Task2": categorize the message according to the type of sexism. Possible categories are: “ideological-inequality”, “stereotyping-dominance”, “objectification”, “sexual-violence” and “misogyny-non-sexual-violence”.

Our detailed analysis began with a sentiment analyzer using a "sexist words" dictionary we created based on our training data. This score corresponded to the number of sexist words identified in each single tweet. Then, we were able to sort our tweets by this score.

Next, we created a Markov Chain to predict words that would follow a certain word. This gives us a sense of the relationships between words and could potentially be used to identify problematic patters in social media users.

We then moved on to develop a predictor for sexism using a Logistic Regression with Matrix Multiplication (Strassen) and Gradient Descent. This gave us the probability (0-1) of a tweet being sexist. With this development we could identify a great number of sexist tweets (1) in our test set just by looking over the resulting probabilities. 

In order to accomplish Logistic Regression with Matrix Multiplication (Strassen), we created a function for Strassen matrix multiplication and then used it within our gradient descent. This recursive function will be very useful when dealing with massive datasets and matrices.

Lastly, we created a tree to construct relationships between sexist words. More specifically, we wanted to find common combinations of words in sexist tweets. For this, we created a co-occurrence matrix for iterating within tweets that had a sexist word already identified.

2. Installation Instructions
The Jupyter Notebook uses a series of libraries. Ensure the following libraries are installed:

import fasttext
import io
import re
import nltk
nltk.download('stopwords')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random 
import csv
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

4. Structure of the notebook

    1. Data exploration and processing

    2. Sexism detector:

        2.1 Dictionary-Based Sentiment Analysis to create sexism score
    
        2.2 Quicksort to organize data by sexism score
    
        2.3 Markov Chains for word predictions
    
        2.4 Algorithm for matrix multiplication (Strassen)
   
        2.5 Logistic regression with Strassen matrix multiplication and Gradient Descent

        2.6 Co-occurrence tree to analyze words commonly used together in sexist tweets
