# Detecting and Predicting Sexist Language in Tweets

## Project Overview

Our project focuses on analyzing sexist bias in tweets to:

1. Predict sexism.
2. Understand relationships between words commonly found together in sexist tweets.

Datasets were prepared by Francisco Rodriguez-Sanchez, Jorge Carrillo-de-Albornoz, Laura Plaza, Julio Gonzalo, Paolo Rosso, Trinidad Donoso, and Miriam Comet for their EXIST 2021 SEXism Identification in Social neTworks Task Guidelines.

### Datasets

The training and test datasets contain the following fields:
- **"Test_case"**: Tag used in the EvALL framework for evaluating classification tasks, set to “EXIST2021” in our case.
- **"Id"**: Unique identifier for each tweet.
- **"Source"**: Social network from which the text was crawled, with values “twitter” and “gab”.
- **"Language"**: Language of the tweet (“en” for English, “es” for Spanish).
- **"Text"**: The text of the tweet.
- **"Task1"**: Indicates whether the tweet is sexist (“sexist”) or not (“non-sexist”).
- **"Task2"**: Categorizes the type of sexism. Possible categories are: “ideological-inequality”, “stereotyping-dominance”, “objectification”, “sexual-violence”, and “misogyny-non-sexual-violence”.

## Methodology

1. **Sentiment Analysis**:
   - Created a dictionary of "sexist words" from training data to analyze the sentiment and identify sexist words in each tweet. Tweets were then sorted by the number of identified sexist words.

2. **Markov Chains**:
   - Developed a Markov Chain model to predict the likelihood of words following specific words. This helps understand word relationships and identify problematic patterns in social media discourse.

3. **Logistic Regression**:
   - Implemented a Logistic Regression model with Strassen Matrix Multiplication and Gradient Descent to predict the probability of a tweet being sexist. This model enabled us to identify a substantial number of sexist tweets based on predicted probabilities.

4. **Strassen Matrix Multiplication**:
   - Created a function for Strassen matrix multiplication to be used in logistic regression. This recursive function enhances efficiency when dealing with large datasets and matrices.

5. **Co-occurrence Tree**:
   - Constructed a tree to analyze common combinations of words in sexist tweets using a co-occurrence matrix.

## Installation Instructions

To run the Jupyter Notebook, ensure the following libraries are installed:

```python
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
