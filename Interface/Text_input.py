import streamlit as st
import pandas as pd
import pickle

from Config import vader, Charts, util
from Interface import widget
from textblob import TextBlob
from annotated_text import annotated_text

model_list = ["Logistic Regression","Multinomial Naive Bayes Classifier"]
model_file_list = [r"Machine_Learning_Model/LR_model.pkl",r"Machine_Learning_Model/MNVBC_model.pkl"]
def text_input():
    raw_text = st.sidebar.text_area("Enter Text Here")
    btn = st.sidebar.button("Analyse")
    if btn:
        text = raw_text.split()
        flag = check_text_count(text)
        if flag == False:
            st.sidebar.error("Error. Please insert more than 3 words in the text input")
        else:
            st.sidebar.success("Execution completed")
            with st.spinner("Generating..."):
                raw_text = util.find_tag_hashtag_word(raw_text)
                raw_text = util.find_url(raw_text)
                raw_text = util.find_date(raw_text)
                raw_text = util.find_sym_word(raw_text)
                raw_text = util.expand_contractions(raw_text)
                word_count = len(raw_text.split())
                token_polarity = vader.analyse_token_sentiment(raw_text)
                vader_polarity = vader.token_polarity(raw_text)
                vader_analysis = vader.categorise_sentiment_text(vader_polarity)
                pos_count = len(vader_analysis[vader_analysis['Sentiment'] == 'positive'].value_counts())
                neg_count = len(vader_analysis[vader_analysis['Sentiment'] == 'negative'].value_counts())
                neu_count = len(vader_analysis[vader_analysis['Sentiment'] == 'neutral'].value_counts())
                
                col1, col2, col3,col4 = st.columns(4)
                col1.metric("Words Count", word_count)
                col2.metric("Positive word Count", pos_count)
                col3.metric("Negative word Count", neg_count)
                col4.metric("Neutral word Count", neu_count)
                st.write('---')
                col5, col6 = st.columns(2)
                with col5:
                    st.write('**Sentiment Result**')
                    tokens = vader.process_text(raw_text)
                    annotated_text(*tokens)
                    st.write('---')
                    st.write('**Token Sentiment Breakdown**')
                    st.write(token_polarity)

                with col6:
                    Charts.Piechart(pos_count, neg_count, neu_count, "Sentiment Types", "650px", "300px")
                    widget.download_but(vader_analysis,'sentiment.csv')

                col7, col8 = st.columns(2)
                with col7:
                    polarity_classification(raw_text)
                with col8:
                    token_sentiment_func(raw_text)
                
                col9, col10 = st.columns(2)
                with col9:
                    ML_train(raw_text)
                with col10:
                    if raw_text is None:
                        st.info("Awaiting for the text input")
                    else:
                        most_common_word = util.most_common_word_func(str(raw_text))
                        most_common_word_dict = most_common_word.to_dict('dict')
                        Charts.barchart(most_common_word_dict, 'Most common words', width="650px", height="500px")
                        widget.download_but(most_common_word, 'Most_common_word.csv')

                        

def check_text_count(text):
    if len(text) < 2:
        return False
    else:
        return True

def ML_train(raw_text):
    predictions = []
    for model in model_file_list:
        filename = model
        model = pickle.load(open(filename, "rb"))
        prediction = model.predict([raw_text])[0]
        predictions.append(prediction)
    
    dict_prediction = {"Models":model_list,"predictions":predictions}
    df = pd.DataFrame(dict_prediction)
    df['prediction_label'] = df['predictions'].apply(lambda x: check_prediction(x))
    log_pred = int(df.iloc[0,2])
    mnvm_pred = int(df.iloc[1,2])
    Charts.ML_pie(log_pred, mnvm_pred)
    widget.download_but(df, 'Machine_learning_prediction_result.csv')

def check_prediction(prediction):
    if prediction == 'Positive':
        return 0
    else:
        return 1

def polarity_classification(raw_text):
    pol_class = TextBlob(raw_text).sentiment
    df = vader.convert_to_df_TextBlob(pol_class)
    dict = df.to_dict()
    metric = list(dict['metric'].values())
    value = list(dict['value'].values())
    Charts.polarity_subjectivity_chart("Polarity and Subjectivity score", value, "650px", "300px")

def token_sentiment_func(raw_text):
    token_polarity = vader.token_polarity(raw_text)
    word = token_polarity.get("Word")
    pol_score = token_polarity.get("Polarity Score")
    Charts.token_sent_barchart(word, pol_score, "Polarity score by token sentiments")






