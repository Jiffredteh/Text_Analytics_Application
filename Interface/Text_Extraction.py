import pandas as pd
import streamlit as st
from Config import Global as glb, util, Charts
from Interface import widget
    
def text_extraction(new_df):
    df = pd.DataFrame(new_df)
    df = df.rename(columns={df.columns[0]: 'Text'})
    df['Text'] = df['Text'].astype("str").astype("string")
    df['Text'] = df['Text'].str.lower()
    glb.flag = True
    util.configure_all(df['Text'], df)
    st.title("Exploratory Data Analysis")
    st.write("**What we found from the uploaded dataset**")
    word_extracation_freq_func(df)

def word_extracation_freq_func(df):
    total_word_count = int(df["Text"].str.split().str.len().sum())
    hashtag_count = int(df["Hashtag(#) and Tag(RT@/@)"].str.split().str.len().sum())
    url_count = int(df["URLs"].str.split().str.len().sum())
    day_count = int(df["Day and Month"].str.split().str.len().sum())
    special_char_count = int(df['Special Chars and Nums'].str.split().str.len().sum())
    cont1 = st.container()
    with cont1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Word Type Extraction**")
            st.metric("Total word count", total_word_count)
            st.metric("Hashtag(#) characters count", hashtag_count)
            st.metric("URLs characters count", url_count)
            st.metric("Punctuations count", special_char_count)
            st.metric("Day & month count", day_count)
            st.caption("The extraction was performed based on the text characteristics")
            widget.download_but(df, 'text_extraction.csv')
        with col2:
            Charts.extraction_chart_func(hashtag_count, url_count, day_count, special_char_count)