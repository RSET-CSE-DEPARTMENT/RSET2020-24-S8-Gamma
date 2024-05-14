import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Download NLTK resources
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    processed_text = ' '.join(words)

    return processed_text

model1 = load_model('/content/drive/MyDrive/model/depression/depressionclassifier.h5')
model2 = load_model('/content/drive/MyDrive/model/anxiety/anxietyclassifier.h5')
model3 = load_model('/content/drive/MyDrive/model/autism/autismclassifier.h5')
model4 = load_model('/content/drive/MyDrive/model/bipolar/bipolarclassifier.h5')
model5 = load_model('/content/drive/MyDrive/model/bpd/bpdclassifier.h5')
model6 = load_model('/content/drive/MyDrive/model/schizophrenia/schizophreniaclassifier.h5')

label_encoder = joblib.load('/content/drive/MyDrive/model/depression/path_to_save_label_encoder.joblib')

word2vec_model = Word2Vec.load('/content/drive/MyDrive/model/depression/w2v.model')

max_post_length = 3601
max_post_length3 = 1461
max_post_length4 = 2490
max_post_length5 = 1996
max_post_length6 = 2490

age = 37
gender = 'female'
location = 'USA'

tweets = [
    "Feeling like I'm stuck in a dark tunnel with no way out. #Depression",
  "Every day feels like a battle against my own mind. #MentalHealth",
  "Trying to find the energy to get out of bed seems impossible right now.",
  "It's like I'm drowning in sadness and I can't seem to catch my breath. #Depression",
  "Feeling completely numb and disconnected from everything around me. #MentalHealth",
  "Why does it feel like everyone else has it all together while I'm barely holding on?",
  "The weight of the world feels like it's crushing me. #Depression",
  "No matter how hard I try, I just can't seem to shake this feeling of emptiness.",
  "Feeling like I'm just going through the motions, with no real purpose or direction. #MentalHealth",
  "Sometimes it feels like there's no light at the end of the tunnel, only darkness."
]
text_corpus = ' '.join(tweets)

processed_text = preprocess_text(text_corpus)

tweet_sequences = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in processed_text.split() if word in word2vec_model.wv]], maxlen=max_post_length)
tweet_sequences3 = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in processed_text.split() if word in word2vec_model.wv]], maxlen=max_post_length3)
tweet_sequences4 = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in processed_text.split() if word in word2vec_model.wv]], maxlen=max_post_length4)
tweet_sequences5 = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in processed_text.split() if word in word2vec_model.wv]], maxlen=max_post_length5)
tweet_sequences6 = pad_sequences([[word2vec_model.wv.key_to_index[word] for word in processed_text.split() if word in word2vec_model.wv]], maxlen=max_post_length6)

tweet_sequences = np.expand_dims(tweet_sequences, axis=-1)

predictions1 = model1.predict(tweet_sequences)
predictions2 = model2.predict(tweet_sequences)
predictions3 = model3.predict(tweet_sequences3)
predictions4 = model4.predict(tweet_sequences4)
predictions5 = model5.predict(tweet_sequences5)
predictions6 = model6.predict(tweet_sequences6)

predicted_labels1 = predictions1
predicted_labels2 = predictions2
predicted_labels3 = predictions3
predicted_labels4 = predictions4
predicted_labels5 = predictions5
predicted_labels6 = predictions6

print(f"Combined Text Corpus: {processed_text}")
print(f"Depression: {predicted_labels1[0]}")
print(f"Anxiety: {predicted_labels2[0]}")
print(f"Autism: {predicted_labels3[0]}")
print(f"Bipolar: {predicted_labels4[0]}")
print(f"BPD: {predicted_labels5[0]}")
print(f"Schizophrenia: {predicted_labels6[0]}")