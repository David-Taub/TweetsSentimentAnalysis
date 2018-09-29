import tqdm
from collections import defaultdict
import keras
import sklearn.utils.class_weight
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GlobalAveragePooling1D, Permute, Activation, Conv1D
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score
import re
from keras import backend as K

TRAIN_CSV_FILEPATH = r'C:\Users\booga\Dropbox\projects\TweetsSentimentAnalysis\data\train.csv'
TEST_CSV_FILEPATH = r'C:\Users\booga\Dropbox\projects\TweetsSentimentAnalysis\data\test.csv'
MODEL_FILEPATH = r'C:\Users\booga\Dropbox\projects\TweetsSentimentAnalysis\model.h5'
VOCAB_SIZE = 10000
BATCH_SIZE = 64


def hashtag_split_func(x):
    return " ".join([a for a in re.split('([A-Z][a-z]+)', x.group(1)) if a])


# string manipulation that removes links, breaks hashtags lower case etc
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

#used to print F1 recall and precision while training, not used for early stop
class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        targ = np.argmax(self.validation_data[1], axis=1)
        print('val_f1 %.3f' % f1_score(targ, predict))
        print('val_recall %.3f' % recall_score(targ, predict))
        print('val_precision %.3f' % precision_score(targ, predict))
        return



def predict_test(tokenizer, max_words_in_tweets):
    test_data_raw = pd.read_csv(TEST_CSV_FILEPATH).values
    test_data = preprocess_data(test_data_raw)
    X_test = tokenizer.texts_to_sequences(test_data[:, 1])
    X_test = pad_sequences(X_test, max_words_in_tweets)
    model = keras.models.load_model(MODEL_FILEPATH)
    predictions = model.predict(X_test)
    return np.argmax(predictions, axis=1)



def build_model(input_length):
    #TODO: use glove embedding
    #TODO: pretrain the net as a LM
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 300, input_length=input_length))
    model.add(SpatialDropout1D(0.5))
    # model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    # model.add(Dense(2, activation='softmax'))

    model.add(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(Conv1D(filters=2, kernel_size=1, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Activation(activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def main():
    train_val_data_raw = pd.read_csv(TRAIN_CSV_FILEPATH).values
    train_val_data = preprocess_data(train_val_data_raw)

    print('Negatives: %d' % train_val_data[train_val_data[:, 2] == 0].size)
    print('Positives: %d' % train_val_data[train_val_data[:, 2] == 1].size)
    print('Baseline accuracy: %.3f' % (train_val_data[train_val_data[:, 2] == 0].size / train_val_data.size))

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, split=' ')
    tokenizer.fit_on_texts(train_val_data[:, 1])

    X = tokenizer.texts_to_sequences(train_val_data[:, 1])
    X = pad_sequences(X)
    Y = to_categorical(train_val_data[:, 2])
    sequence_words_amount = X.shape[1]

    model = build_model(sequence_words_amount)
    # class_weight used to balance type I and type II errors due to imbalanced dataset
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',
                                                                   np.unique(train_val_data[:, 2]),
                                                                   train_val_data[:, 2])
    callbacks = [EarlyStopping(patience=3),
                 ModelCheckpoint(MODEL_FILEPATH, save_best_only=True),
                 Metrics()]
    # model.fit(X, Y,
    #           validation_split=0.25,
    #           epochs=50,
    #           batch_size=BATCH_SIZE,
    #           callbacks=callbacks,
    #           class_weight=class_weight)
    print(predict_test(tokenizer, sequence_words_amount))
    print(get_significant_words(X, tokenizer))


def get_significant_words(X, tokenizer, n=100):
    model = keras.models.load_model(MODEL_FILEPATH)
    word_to_score = defaultdict(list)
    foo = K.function([model.input, K.learning_phase()], [model.layers[-3].output])
    # inverse tokenizer words
    ind_to_word = {v: k for k, v in tokenizer.word_index.items()}
    ind_to_word[0] = 'N/A'
    for t in tqdm.tqdm(range(X.shape[0] // BATCH_SIZE)):
        start = t * BATCH_SIZE
        end = min((t + 1) * BATCH_SIZE, X.shape[0])
        seqs = X[start:end, :]
        out = foo([seqs, 0.0])[0]
        scores = out[:, :, 1] - out[:, :, 0]
        for i in range(seqs.shape[0]):
            for j in range(seqs.shape[1]):
                word = ind_to_word[seqs[i, j]]
                word_to_score[word].append(scores[i, j])
    # avg word_to_score
    word_to_avg_score = {k: np.mean(v) for k, v in word_to_score.items()}
    sorted_word_avg_score = sorted(word_to_avg_score.items(), key=lambda x: x[1])
    return sorted_word_avg_score[-n:]




if __name__ == '__main__':
    main()
