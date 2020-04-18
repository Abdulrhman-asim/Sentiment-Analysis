import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import tkinter as tk
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


def preprocessing():

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    tweets = pd.read_csv('Tweets.csv')
    tweets = tweets[['text', 'airline_sentiment']]
    x = []
    y = []

    # tweets.sort_values(inplace = True, by = ['airline_sentiment'])
    # tweets = tweets.reset_index(drop=True)
    sents = []

    for i in range(len(tweets)):
        tweets.loc[i,'text'] = tweets.loc[i,'text'].lower()
        words = tweets.loc[i,'text'].split()

        fltrdWords = []

        for w in words:
            w = w.strip()
            if w[0] == '@':
                # print(w)
                continue
            fltrdWords.append(w)
        sents.append(fltrdWords)

    featureModel = Word2Vec(sentences=sents, size=100, window=5, min_count=2, workers=4, sg=1)

    for i in range(len(tweets)):

        temp = tweets.loc[i,'airline_sentiment']
        text = tweets.loc[i,'text'].split()
        fv = np.zeros(100)

        if temp =='neutral':
            temp = 2
        elif temp == 'positive':
            temp = 1
        elif temp == 'negative':
            temp = 0
        else:
            temp = -1

        for word in text:
            curFv = None
            if word in featureModel.wv.vocab:
                curFv = np.array(featureModel.wv[word])
            else:
                curFv =np.zeros(100)
            fv = fv + curFv

        fv = np.divide(fv, len(text))

        x.append(fv)
        y.append(temp)

    return x, y, featureModel


x,y,features = preprocessing()

x = np.array(x).reshape(-1, 1, 100, 1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.75)

model = Sequential()

model.add(Flatten(input_shape=x.shape[1:]))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, validation_split=0.2, epochs=50)

predictions = model.predict_classes(X_test)
acc = accuracy_score(y_true=Y_test, y_pred=predictions)
print('=======================')
print("Accuracy: " + str(acc))

# textt = 'Thank you so much for such a comfortable journey'
#
# # 'omg i can\'t believe the flight has been delayed FOR THE THIRD TIME @unitedairlines get your shit together'
def sentVectorize(text):
    text = str(text.get())
    text = text.lower()
    text = text.split()

    fv = np.zeros(100)
    for word in text:
        curFv = None
        if word in features.wv.vocab:
            curFv = np.array(features.wv[word])
        else:
            curFv = np.zeros(100)
        fv = fv + curFv

    fv = np.divide(fv, len(text))
    fv = np.array(fv).reshape(-1, 1, 100, 1)
    pred = model.predict(fv)
    pred = list(pred[0])
    ans = pred.index(max(pred))

    txtbx.delete("1.0", "end")
    if ans == 1:
        txtbx.insert(tk.END, 'Positive Review')
    if ans == 0:
        txtbx.insert(tk.END, 'Negative Review')
    if ans == 2:
        txtbx.insert(tk.END, 'Neutral Review')


master = tk.Tk()
master.geometry("800x400")
master.title('Sentiment Analysis')

entryCheck = tk.StringVar()

inpt = tk.Entry(master, textvariable = entryCheck , width = 80)
inpt.pack()

txtbx = tk.Text(master, height=1, width=20)
txtbx.pack()

btn = tk.Button(text="Sumbit", width=30,
                          command=lambda: sentVectorize(entryCheck))
btn.pack()

tk.mainloop()



