import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib

from statistics import mean
from Config import vader, util, Charts
from Interface import widget


def sent_dashboard(data):
    df = pd.DataFrame(data)
    df['vader_polarity'] = df['Cleaned_Text'].apply(vader.vadar_sentiment)
    df['vader_analysis'] = df['vader_polarity'].apply(vader.categorise_sentiment)

    # Apply Vader sentiment analysis
    st.title("Sentiment Analysis")
    st.sidebar.write('---')
    st.sidebar.header("Sentiment Analysis")
    st.sidebar.markdown("Configuration")
    option = st.sidebar.selectbox("Pick one sentiment", ['positive', 'negative', 'neutral'])
    st.sidebar.caption("the visualisations will be displayed based on the option applied.")
    sentiment_class_func(df)
    st.write('---')
    sentiment_polarity_func(df)
    # st.write('---')
    # sentiment_sent_func(df,option)
    st.write('---')
    # sentiment_words_frequency_func(df,option)
    st.write('---')
    positive_negative_sentiment_freq_func(df)
    st.write('---')
    # if st.sidebar.checkbox("Data Modelling"):
    st.sidebar.write('---')
    util.data_partition(df)


def sentiment_class_func(df):
    df = df[['Cleaned_Text', 'vader_polarity', 'vader_analysis']]
    st.header("Summary at a glance")
    st.subheader("**Sentiment type**")
    col1, col2, col3, col4 = st.columns(4)
    # with col1:
    total = len(df)
    positive = len(df[df['vader_analysis'] == 'positive'])
    negative = len(df[df['vader_analysis'] == 'negative'])
    neutral = len(df[df['vader_analysis'] == 'neutral'])
    col1.metric("Total Observation", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)
    col4.metric("Neutral", neutral)
    # with col3:
    col5, col6 = st.columns(2)
    with col5:
        util.AgGridTable(df)
    with col6:
        Charts.Piechart(positive, negative, neutral, "Text Type Extraction", "", "600px")


def sentiment_polarity_func(df):
    st.subheader('**Polarity score**')
    col5, col6, col7, col8, col9 = st.columns(5)
    overall_mean = round(mean(df['vader_polarity']), 3)
    positive_mean = round(mean(df['vader_polarity'] > 0), 3)
    negative_mean = round(mean(df['vader_polarity'] < 0), 3)
    highest_score = df['vader_polarity'].max()
    lowest_score = df['vader_polarity'].min()
    col5.metric("General Average polarity score", overall_mean)
    col6.metric("General Highest polarity score", highest_score)
    col7.metric("General Lowest polarity score", lowest_score)
    col8.metric("Average positive score", positive_mean)
    col9.metric("Average negative score", negative_mean)
    col1, col2 = st.columns(2)
    with col1:
        fig_2 = px.histogram(df, x="vader_polarity", nbins=80)
        fig_2.update_traces(marker_color='Navy')
        fig_2.update_layout(title_text='Polarity score classification',
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width = 660)
        st.plotly_chart(fig_2)
    with col2:
        fig_3 = px.histogram(df, x="vader_polarity",
                             color="vader_analysis", nbins=80)
        fig_3.update_layout(title_text='Polarity score by sentiment types',
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width = 660)
        st.plotly_chart(fig_3)
    widget.download_but(df, 'sentiment_analysis.csv')

def sentiment_words_frequency_func(df,option):
    st.header("Word Frequency Distribution by sentiment types")
    st.subheader("**Bar Chart**")
    comment = df[(df['vader_analysis'] == option)]['Cleaned_Text']
    comment_text_corpus = comment.str.cat(sep=' ')
    comment_most_common = util.most_common_word_func(comment_text_corpus)
    comment_df = comment_most_common.head(25)
    comment_df_dict = comment_df.to_dict('dict')
    Charts.barchart(comment_df_dict, option, width=1400, height=700)
    widget.download_but(comment_df, option+".csv")
    if 'positive' in option:
        Charts.plotly_wordcloud(comment_text_corpus, matplotlib.cm.cool)
    if 'negative' in option:
        Charts.plotly_wordcloud(comment_text_corpus, matplotlib.cm.seismic)
    if 'neutral' in option:
        Charts.plotly_wordcloud(comment_text_corpus, matplotlib.cm.cividis)


def positive_negative_sentiment_freq_func(df):
    st.subheader('Word Frequency Analysis by Positive and Negative Sentiments')
    vader_positive_comment = df[(
        df['vader_analysis'] == 'positive')]['Cleaned_Text']
    vader_positive_comment_text_corpus = vader_positive_comment.str.cat(
        sep=' ')
    vader_positive_comment_most_common = util.most_common_word_func(
        vader_positive_comment_text_corpus)
    vader_positive_comment_df = vader_positive_comment_most_common
    vader_positive_comment_df = vader_positive_comment_df.rename(
        columns={'Frequency': 'Positive Frequency'})

    vader_negative_comment = df[(
        df['vader_analysis'] == 'negative')]['Cleaned_Text']
    vader_negative_comment_text_corpus = vader_negative_comment.str.cat(
        sep=' ')
    vader_negative_comment_most_common = util.most_common_word_func(
        vader_negative_comment_text_corpus)
    vader_negative_comment_df = vader_negative_comment_most_common
    vader_negative_comment_df = vader_negative_comment_df.rename(
        columns={'Frequency': 'Negative Frequency'})
    merge = pd.merge(vader_positive_comment_df,
                     vader_negative_comment_df, on='Word')
    merge['Negative Frequency'] *= -1
    merge = merge.sort_values(by='Negative Frequency', ascending=True)
    # option = st.slider("Select number of words to display",10, 25, key='pos_neg_sent')
    chart = merge.head(25)
    dict = chart.to_dict('dict')
    Charts.post_neg_word_freq_chart(dict, 'Positive and Negative Sentiment')
    widget.download_but(merge, 'Positive_&_Negative_sentiment.csv')
