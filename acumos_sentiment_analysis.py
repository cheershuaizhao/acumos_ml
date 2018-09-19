import numpy as np
import seaborn as sn
import pandas as pd
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.utils import plot_model


from keras.utils.data_utils import get_file
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np

embed_size = 50 # max embedding length for a word
max_length = 200 # max number of words in a review

path = get_file('imdb_full.pkl',
               origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                md5_hash='d091312047c43cf9e4e38fef92437263')
f = open(path, 'rb')
(x_train, y_train), (x_test, y_test) = pickle.load(f)

word_to_id = imdb.get_word_index()
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
x_test = [' '.join(id_to_word[index] for index in data) for data in x_test]
x_train = [' '.join(id_to_word[index] for index in data) for data in x_train]

X = x_train + x_test
y = y_train + y_test


t = Tokenizer()
t.fit_on_texts(x_train)
vocab_size = len(t.word_index) + 1

x_train = t.texts_to_sequences(x_train)
padded_docs = pad_sequences(x_train, maxlen=max_length, padding='post')

#use pre-trained Google word embedding
embeddings_index = dict()
f = open('/home/shuai/repo/toxic comment classfication/glove.6B/glove.6B.50d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, embed_size))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector



#build the model
from keras.models import Model as KerasModel
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, concatenate, add, RepeatVector, Permute
from keras.layers import Flatten, Merge, multiply, Lambda
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras import backend as K


#RCNN model
inp1 = Input(shape=(max_length,))
x = Embedding(vocab_size, embed_size, weights=[embedding_matrix])(inp1)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
x = Dense(100, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
rcnn_model = KerasModel(inputs=inp1, outputs=x)
rcnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(rcnn_model.summary())
plot_model(rcnn_model, to_file='rcnn_model.png')


#train the model
earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
rcnn_model.fit(padded_docs, y_train, batch_size=128, epochs=10, verbose=1, callbacks=[earlyStopping],
      validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
      
#rcnn_model.save('rcnn_model.h5')

from acumos.modeling import  create_namedtuple, List

Input = create_namedtuple('Input', [('review', List[str]), ('g_truth', List[float])])
Output = create_namedtuple('Output', [('sentiment', List[float])])

