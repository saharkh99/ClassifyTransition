import re
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


def RemoveStopWord(text):
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

negative=[['sorry','unable'],['sorry','terminate'],['unfortunately']]
positives=[['congratulations','success'],['pleased','accept'],['delightful','great'],['pleasure'],['congratulations'],['accept','gift']],

if __name__ == '__main__':
    df = pd.read_excel('dataset.xls')
    df2=pd.read_excel('interactions.xls')

    q1 = df["Close_Value"].quantile(0.25)
    q3 = df["Close_Value"].quantile(0.75)
    IQR = q3 - q1
    df = df[(df["Close_Value"] > (q1 - (1.5 * IQR))) & (df["Close_Value"] < (q3 + (1.5 * IQR)))]
    df.drop(['Created Date', 'Close Date'], inplace=True, axis=1, errors='ignore')
    df = df[df['Stage'] != 'In Progress']
    df['is_won'] = df['Stage'].apply(lambda x: 1 if x=='Won' else 0)
    # plot bar for product/value
    fig, ax = plt.subplots()
    ax.scatter(df['is_won'], df['Product'])
    pd.concat([df.groupby('Product')['Close_Value'].sum()],axis=1).plot.bar()
    plt.show()
    df2.drop(['InteractionDate', 'InteractionType(Call/Email/SMS)'], inplace=True, axis=1, errors='ignore')
    df2['Extracted Interaction Text'] = df2['Extracted Interaction Text'].apply(lambda x: RemoveStopWord(x))
    # print(Counter(df['Extracted Interaction Text'].apply(lambda x: str(x))).most_common(1000))
    df2.columns = ['interactionID', 'fromEmailId', 'ContactEmailID', 'Extracted Interaction Text']
    # print(Counter(" ".join(wordCount(df["Extracted Interaction Text"])).split()).most_common(1000))
    #  find positive or negative emails
    for i in negative:
        for j in i:
            df2['feeling'] = df2['Extracted Interaction Text'].apply(lambda x: 'n' if j in x else 'f')
    for i in positives:
        df2['feeling'] = df2['Extracted Interaction Text'].apply(lambda x: 'p' if j in x else 'f')
    #  merge two dataset
    df['ContactEmailID'] = pd.merge(df, df2, on=['ContactEmailID'], how='inner').all(axis=1)
    df['feeling']=df2['feeling']
    number = preprocessing.LabelEncoder()
    df["Product"] = number.fit_transform(df["Product"])
    df["feeling"] = number.fit_transform(df["feeling"])
    df['Agent'] = number.fit_transform(df["Agent"])
    df['is_won'] = number.fit_transform(df["is_won"])
    df['Close_Value'] = number.fit_transform(df["Close_Value"])
    data = df.fillna(-999)
    X = data.filter([
        "Product",
        "Close_Value",
        "feeling",
        "Agent",
    ])
    # random forest model
    Y = data["is_won"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    RFRegressor = RandomForestClassifier(n_estimators=20)
    RFRegressor.fit(X_train, Y_train)
    Y_PRED = RFRegressor.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_PRED))
    print(metrics.accuracy_score(Y_test,Y_PRED))
    print(metrics.f1_score(Y_test,Y_PRED))
    print(metrics.confusion_matrix(Y_test,Y_PRED))





