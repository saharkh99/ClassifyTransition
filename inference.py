#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import re

from pandas import DataFrame
from joblib import load
import string
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score, classification_report, plot_confusion_matrix

import numpy as np
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def words_in_str(big_long_string, search_list):
    counter = 0
    for word in big_long_string.split(" "):
        if word in search_list: counter += 1
        if counter == len(search_list): return True
    return False


def wordCount(text):
    try:
        text = text.lower()
        regex = re.compile("[" + re.escape(string.punctuation) + "0-9\\r\\t\\n]")
        txt = regex.sub(" ", text)
        words = [
            w
            for w in txt.split(" ")
            if w not in ENGLISH_STOP_WORDS and len(w) > 3
        ]
        return str(words)
    except Exception:
        return 0


BEST_MODEL_PATH = "resources/best_model.joblib"  # change this line as you wish

# model = load(BEST_MODEL_PATH)
negative = [['sorry', 'unable'], ['sorry', 'terminate'], ['unfortunately']]
positives = [['congratulations', 'success'], ['pleased', 'accept'], ['delightful', 'great'], ['pleasure'],
             ['congratulations'], ['accept', 'gift']],


def inference(path: str) -> [int]:
    '''
    path: a DataFrame
    result is the output of function which should be 
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    '''
    result = []
    df = pd.read_excel('dataset.xls')
    df2 = pd.read_excel('interactions.xls')
    df3 = pd.read_excel('dataset.xls')
    q1 = df["Close_Value"].quantile(0.25)
    q3 = df["Close_Value"].quantile(0.75)
    IQR = q3 - q1
    df = df[(df["Close_Value"] > (q1 - (1.5 * IQR))) & (df["Close_Value"] < (q3 + (1.5 * IQR)))]
    df.drop(['Created Date', 'Close Date'], inplace=True, axis=1, errors='ignore')
    df3.drop(['Created Date', 'Close Date'], inplace=True, axis=1, errors='ignore')
    df = df[df['Stage'] != 'In Progress']
    df['is_won'] = df['Stage'].apply(lambda x: 1 if x == 'Won' else 0)

    df2.drop(['InteractionDate', 'InteractionType(Call/Email/SMS)'], inplace=True, axis=1, errors='ignore')
    df2['Extracted Interaction Text'] = df2['Extracted Interaction Text'].apply(lambda x: wordCount(x))
    df2.columns = ['interactionID', 'fromEmailId', 'ContactEmailID', 'Extracted Interaction Text']
    for i in negative:
        for j in i:
            df2['feeling'] = df2['Extracted Interaction Text'].apply(lambda x: 'n' if j in x else 'f')
    for i in positives:
        df2['feeling'] = df2['Extracted Interaction Text'].apply(lambda x: 'p' if j in x else 'f')
    # ///////////////////////////////////////////////////////////////
    df['ContactEmailID'] = pd.merge(df, df2, on=['ContactEmailID'], how='inner').all(axis=1)
    df3['ContactEmailID'] = pd.merge(df3, df2, on=['ContactEmailID'], how='inner').all(axis=1)
    df['feeling'] = df2['feeling']
    df3['feeling'] = df2['feeling']
    number = preprocessing.LabelEncoder()
    df["Product"] = number.fit_transform(df["Product"])
    df3["Product"] = number.fit_transform(df3["Product"])
    df["feeling"] = number.fit_transform(df["feeling"])
    df3["feeling"] = number.fit_transform(df3["feeling"])
    # df['Customer'] = number.fit_transform(df["Customer"])
    df['Agent'] = number.fit_transform(df["Agent"])
    df3['Agent'] = number.fit_transform(df3["Agent"])
    df['is_won'] = number.fit_transform(df["is_won"])
    df['Close_Value'] = number.fit_transform(df["Close_Value"])
    df3['Close_Value'] = number.fit_transform(df3["Close_Value"])

    data = df.fillna(-999)
    X = data.filter([
        "Product",
        # "Customer",
        "Close_Value",
        "feeling",
        "Agent",
    ])
    Y = data["is_won"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    RFRegressor = RandomForestClassifier(n_estimators=20)
    RFRegressor.fit(X_train, Y_train)
    data = df3.fillna(-999)
    X3 = data.filter([
        "Product",
        # "Customer",
        "Close_Value",
        "feeling",
        "Agent",
    ])
    result = RFRegressor.predict(X3)
    print(result)
    return result
if __name__ == '__main__':
  inference('dataset.xls')
