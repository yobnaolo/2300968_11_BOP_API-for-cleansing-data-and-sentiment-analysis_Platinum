from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
import re,string
from services.cleansing import cleansing
from fastapi import status
from fastapi.responses import PlainTextResponse
from fastapi.responses import Response
import sqlite3
import pandas as pd
from services.database import input_database
from utils.consume_model import get_sentiment_result


pretrained = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

async def insert_db(df):
    """
    Insert data to database
    :param df: dataframe
    :return: connection
    """
    conn = sqlite3.connect('tweets.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tweets
                        (Tweet TEXT, Tweets_clean TEXT, Sentiment TEXT)''')
    df.to_sql('tweets', conn, if_exists='append', index=False)
    conn.close()


async def get_sentiment(text, model_type):
    """
    Get sentiment
    :param text: user input with string type
    :return: sentiment result
    """
    result = await cleansing(text)
    result = result['data']

    if model_type == "huggingface":
        sentiment = classifier(result)
    elif model_type == "nn":
        sentiment = await get_sentiment_result(result, model_type)
    elif model_type == "rnn":
        sentiment = await get_sentiment_result(result, model_type)
    elif model_type == "lstm":
        sentiment = await get_sentiment_result(result, model_type)

    content  = {
        "ok" : True,
        "code" : status.HTTP_200_OK,
        "data" : {
            "data" : result,
            "sentiment" : sentiment
            },
        "messege" : "Success"
    }

    return content

async def get_sentiment_file(df):
    tweets_list = df.iloc[:,0].tolist()
    tweets_clean_list = []
    sentiment_list = []

    for i in tweets_list:
        tweet_clean = await cleansing(i)
        result = tweet_clean['data']
        tweets_clean_list.append(result)

        sentiment = classifier(result)
        sentiment_list.append(sentiment[0]['label'])

    df['Tweets_clean'] = tweets_clean_list
    df['Sentiment'] = sentiment_list

    await input_database(df=df)
    
    csv_data = df.to_csv(index=False)
    response = Response(content = csv_data, media_type="text/csv")
    response.headers ["Content-Disposition"] = "attachment ; filename=data.csv"
    return response