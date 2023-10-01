from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
from services.cleansing import cleanse_text
from sklearn.feature_extraction.text import CountVectorizer

async def get_sentiment_result(input_text, model_type):
    if model_type == "rnn" or model_type == "lstm":
        total_data, labels = await generate_tokenizer()
        max_features = 100000
        tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
        tokenizer.fit_on_texts(total_data)

        sentiment = ['negative','positive']

        text = [input_text]
        predicted = tokenizer.texts_to_sequences(text)
        guess = pad_sequences(predicted, maxlen=96)

        if model_type == "rnn":
            model = load_model('model_version_control/model_rnn.h5')
        elif model_type == "lstm":
            model = load_model('model_version_control/lstm_model.h5')

        prediction= model.predict(guess)
        polarity = np.argmax(prediction[0])
        return sentiment[polarity]
    else :
        df = pd.read_csv("utils/data/sentiment_data.csv")
        data_preprocessed = df.text.tolist()

        loaded_model = pickle.load(open("model_version_control/model.p", "rb"))

        count_vect = CountVectorizer()
        count_vect.fit(data_preprocessed)
        text_clean = await cleanse_text(input_text)
        text = count_vect.transform([text_clean])
        result = loaded_model.predict(text)[0]

        return result


async def generate_tokenizer():
    df = pd.read_csv("utils/data/sentiment_data.csv")
    neg = df.loc[df['label'] == 'negative'].text.tolist()
    pos = df.loc[df['label'] == 'positive'].text.tolist()
    net = df.loc[df['label'] == 'neutral'].text.tolist()

    neg_label = df.loc[df['label'] == 'negative'].label.tolist()
    pos_label = df.loc[df['label'] == 'positive'].label.tolist()
    net_label = df.loc[df['label'] == 'neutral'].label.tolist()

    total_data = pos + neg + net
    labels = pos_label + neg_label + net_label

    return total_data, labels
