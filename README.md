
# Wine_Reviews_Text_Mining_Analytics
![enter image description here](https://edenvaleinn.com/wp-content/uploads/2019/05/apple-hill-wine-tasting-tours-slide.jpg)
This project evaluates wine review descriptions with a variety of different text mining tools, including a sentiment analysis, word cloud visual (TextBlob), and an NFM &amp; LDA model. It provides statistical analysis using a Logistic regression and offers a recommendation system using the KNN model.

## Steps
1. Import necessary packages
2. Examining descriptive stats of the dataset
3. Clean Data Appropriately 
4. Execute Exploratory Data analysis 
5. Conduct Topic Modeling (LDA, NFM, Gensim Model)
6. Calculate Sentiment scores (TextBlob)
7. Perform Statistical Models (Logistic Regression, KNN Recommendation System)

## Requirements

* Python
* Google Colab

## Packages 

import pandas as pd

import numpy as np

  

import seaborn as sns

import matplotlib.pyplot as plt

  

import re

import string

  

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF

  

from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob

import nltk

nltk.download('punkt')

nltk.download('stopwords')

  

import spacy #used spacy for text prepocessing

  

import gensim

from gensim import corpora

  

#### libraries for visualization

import pyLDAvis

import pyLDAvis.gensim_models as gensimvis

%matplotlib inline

## Launch

Download the zipped dataset "*Wine_Review_Data.csv* ", unzip it, and upload it to a specified location in Google Drive.
Download *Wine_Entusiast_Review_Analytics.ipynb* and open the file in Google Colab.
Change the File path to the location of the file in your Google Drive.

Run the code. 

## Authors

[Joshua Rotuna](https://github.com/joshrotuna)

[Silvia Ji](https://github.com/jisilvia)

## License

This project is licensed under the  [MIT](https://choosealicense.com/licenses/mit/)  License.

## Acknowledgements

The project files were provided by Nohel Zaman, Loyola Marymount University.
