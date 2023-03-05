import pandas as pd
import numpy as np
from statistics import mean
import streamlit as st
from Config import Global as glb, util, Charts
from Interface import widget, sentiment
from streamlit_echarts import st_echarts

def text_cleaning_dashboard(df):
    st.title("Natural Language Pre-processing")
    st.write("**What we cleaned from the original dataset**")
    st.write("Summary at a glance of the processed dataset")
    word_frequency_func(df)
    st.write('---')
    most_common_word_frequency(df)

def text_cleaning(new_df):
    df = pd.DataFrame(new_df)
    df = df.rename(columns={df.columns[0]: 'Text'})
    df['Text'] = df['Text'].astype("str").astype("string")
    df['Text'] = df['Text'].str.lower()
    glb.flag = False
    @st.cache_data
    util.configure_all(df['Text'], df)
    text_cleaning_dashboard(df)
    st.write('---') 
    if st.sidebar.checkbox("Sentiment analysis"):
        sentiment.sent_dashboard(df)

def word_frequency_func(df):
    # Apply aggregate function on the words
    text_count = int(df["Text"].str.split().str.len().sum())
    text_mean = int(mean(df['Text'].apply(util.word_count_func)))
    text_max = int(max(df['Text'].apply(util.word_count_func)))
    cleaned_text_count = int(df['Cleaned_Text'].str.split().str.len().sum())
    cleaned_text_mean = int(mean(df['Cleaned_Text'].apply(util.word_count_func)))
    cleaned_text_max = int(max(df['Cleaned_Text'].apply(util.word_count_func)))

    total_word_diff = cleaned_text_count - text_count
    x = np.float32(total_word_diff)
    total_word_diff = x.item()

    mean_word_diff = cleaned_text_mean - text_mean
    mean_word_diff = round(mean_word_diff)
    y = np.float32(mean_word_diff)
    mean_word_diff = y.item()

    max_word_diff = cleaned_text_max - text_max
    z = np.float32(max_word_diff)
    max_word_diff = z.item()
    col1, col2, col3 = st.columns(3)
    with col1:
        col1.metric("Total words count", cleaned_text_count, total_word_diff)
        Charts.nlp_barchart('Total words', text_count, cleaned_text_count, 400, 300)
    with col2:
        col2.metric("Average words count", cleaned_text_mean, mean_word_diff)
        Charts.nlp_barchart('Average words', text_mean, cleaned_text_mean, 400, 300)
    with col3:
        col3.metric("Maximum text length", cleaned_text_max, max_word_diff)
        Charts.nlp_barchart('Maximum text length', text_max, cleaned_text_max, 400, 300)
    
    widget.download_but(df, 'cleaned_data.csv')

def most_common_word_frequency(df):
    original_text_corpus = df['Text'].str.cat(sep=' ')
    cleaned_text_corpus = df['Cleaned_Text'].str.cat(sep=' ')
    cleaned_text_most_common_word = util.most_common_word_func(cleaned_text_corpus)
    original_text_most_common_word = util.most_common_word_func(original_text_corpus)
    st.subheader("Most common words")
    col1, col2 = st.columns(2)
    with col1:
        original_text_most_common_word_df = original_text_most_common_word.head(25)
        original_text_most_common_word_dict = original_text_most_common_word_df.to_dict('dict')
        Charts.barchart(original_text_most_common_word_dict, 'Original words', width=650, height=500)
        widget.download_but(original_text_most_common_word_df, "ori_most_common_word.csv")
    with col2:
        cleaned_text_most_common_word_df = cleaned_text_most_common_word.head(25)
        cleaned_text_most_common_word_dict = cleaned_text_most_common_word_df.to_dict('dict')
        Charts.barchart(cleaned_text_most_common_word_dict, 'Processed words', width=650, height=500)
        widget.download_but(cleaned_text_most_common_word_df, "processed_most_common_word.csv")
