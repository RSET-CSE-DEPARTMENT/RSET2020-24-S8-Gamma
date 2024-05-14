import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv('/content/drive/MyDrive/model/depression/data_dep.csv')

df = df.dropna(subset=['processed_post'])

X_train = [str(post).split() for post in df['processed_post']]  # Convert to string before splitting

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df['label'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

word2vec_model = Word2Vec.load('/content/drive/MyDrive/model/depression/w2v.model')

embedding_dim = word2vec_model.vector_size

max_post_length = max(len(post) for post in X_train)

embedding_matrix = np.zeros((len(word2vec_model.wv.key_to_index) + 1, embedding_dim))
for word, i in word2vec_model.wv.key_to_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

X_train_padded = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in post if word in word2vec_model.wv] for post in X_train], maxlen=max_post_length)
X_val_padded = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in post if word in word2vec_model.wv] for post in X_val], maxlen=max_post_length)

print("Class distribution before SMOTE:", Counter(y_train))

# Define the CNN model
model = Sequential()

embedding_layer = Embedding(
    input_dim=len(word2vec_model.wv.key_to_index) + 1,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_post_length,
    trainable=False
)
model.add(embedding_layer)

# Convolutional layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(Dropout(0.25))

# Max-pooling layer
model.add(MaxPooling1D(pool_size=2))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.summary()

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

epochs = 10
history = model.fit(
    X_train_padded, np.array(y_train),
    epochs=epochs, batch_size=64,
    validation_data=(X_val_padded, np.array(y_val)),
    callbacks=[early_stopping]  
)

#Save the model
model.save('/content/drive/MyDrive/model/depression/depressionclassifier.h5')

import joblib
joblib.dump(label_encoder, '/content/drive/MyDrive/model/depression/path_to_save_label_encoder.joblib')