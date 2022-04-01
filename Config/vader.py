import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st

## Sentiment Analysis - Vader
sent_i = SentimentIntensityAnalyzer()
def vadar_sentiment(text):
    """ Calculate and return the nltk vadar (lexicon method) sentiment """
    return sent_i.polarity_scores(text)['compound']

def process_text(doc):
    tokens = []
    for token in doc.split():
        res = sent_i.polarity_scores(token)['compound']
        if res > 0.1:
            tokens.append((token,"Positive","#afa"))
        elif res <= -0.1:
            tokens.append((token,"Negative","#faa"))
        else:
            tokens.append((" " + token + " "))
    return tokens
        

def analyse_token_sentiment(docx):
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = sent_i.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    result = {'positive': pos_list, 'negative': neg_list}
    return result

def token_polarity(docx):
    my_dict = {"Word":[],"Polarity Score":[]}
    for i in docx.split():
        res = sent_i.polarity_scores(i)['compound']
        my_dict["Word"].append(i)
        my_dict["Polarity Score"].append(res)
    return my_dict

def categorise_sentiment_text(sentiment, neg_threshold=-0.05, pos_threshold=0.05):
    # categorise the sentiment value as positive (1), negative (-1) 
    # or neutral (0) based on given thresholds '''
    # st.write(sentiment)
    def sent_label(pol_score):
        if pol_score < neg_threshold:
            return 'negative'
        elif pol_score > pos_threshold:
            return 'positive'
        else:
            return 'neutral'
    word = sentiment.get("Word")
    pol_score = sentiment.get("Polarity Score")
    df = pd.DataFrame({'Word': word,'Polarity Score': pol_score})
    df['Sentiment'] = df['Polarity Score'].apply(lambda x: sent_label(x))
    return df

def categorise_sentiment(sentiment, neg_threshold=-0.05, pos_threshold=0.05):
    # categorise the sentiment value as positive (1), negative (-1) 
    # or neutral (0) based on given thresholds '''
    if sentiment < neg_threshold:
        label = 'negative'
    elif sentiment > pos_threshold:
        label = 'positive'
    else:
        label = 'neutral'
    return label

# Sentiment to Dataframe (Vader)
def convert_to_df_Vadar(vader_sentiment, vader_analysis):
    vader_dict = {'Polarity': [vader_sentiment], 'Sentiments': [vader_analysis]}
    vader_df = pd.DataFrame(vader_dict)
    return vader_df

def convert_to_df_TextBlob(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity,'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric','value'])
    return sentiment_df