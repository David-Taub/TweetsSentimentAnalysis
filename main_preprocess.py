import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

TRAIN_CSV_FILEPATH = r'C:\Users\booga\Dropbox\projects\TweetsSentimentAnalysis\data\train.csv'
TEST_CSV_FILEPATH = r'C:\Users\booga\Dropbox\projects\TweetsSentimentAnalysis\data\test.csv'


def hashtag_split_func(x):
    return " ".join([a for a in re.split('([A-Z][a-z]+)', x.group(1)) if a])


def preprocess_data(data_raw):
    data = data_raw
    # split hashtags
    data[:, 1] = np.vectorize(lambda x: re.sub('[#|@](\w+)', hashtag_split_func, x))(data[:, 1])
    # make lower case
    data[:, 1] = np.vectorize(lambda x: x.lower())(data[:, 1])
    # remove links
    data[:, 1] = np.vectorize(lambda x: re.sub('http\S+', '', x))(data[:, 1])
    data[:, 1] = np.vectorize(lambda x: re.sub('\S+://\S+', '', x))(data[:, 1])
    # remove non alpha-numeric and space-like
    data[:, 1] = np.vectorize(lambda x: re.sub('[^0-9a-zA-Z\s\']', '', x))(data[:, 1])
    # replace space-like with spaces
    data[:, 1] = np.vectorize(lambda x: re.sub('[\s]+', ' ', x))(data[:, 1])
    # strip
    data[:, 1] = np.vectorize(lambda x: x.strip())(data[:, 1])

    # remove empty text rows
    data = data[np.vectorize(lambda x: len(x) > 0)(data[:, 1]), :]
    return data


train_val_data_raw = pd.read_csv(TRAIN_CSV_FILEPATH).values
train_val_data = preprocess_data(train_val_data_raw)

print(train_val_data[train_val_data[:, 2] == 0].size)
print(train_val_data[train_val_data[:, 2] == 1].size)
print(train_val_data[train_val_data[:, 2] == 0].size / train_val_data.size)

VOCAB_SIZE = 30000
tokenizer = Tokenizer(num_words=VOCAB_SIZE, split=' ')
tokenizer.fit_on_texts(train_val_data[:, 1])
X = tokenizer.texts_to_sequences(train_val_data[:, 1])
X = pad_sequences(X)
Y = to_categorical(train_val_data[:, 2])



EMBEDDED_DIM = 500
LSTM_OUT = 196
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDED_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, validation_split=0.15, epochs=50, batch_size=32)


# test_data_raw = pd.read_csv(TEST_CSV_FILEPATH).values
# test_data = preprocess_data(test_data_raw)
# X_test = tokenizer.texts_to_sequences(test_data[:, 1])
# X_test = pad_sequences(X_test, train_val_data.shape[1])
