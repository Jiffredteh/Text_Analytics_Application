import streamlit as st
import pandas as pd
import plotly.express as px
from Config import Data_Access as da, util
from Interface import Text_Extraction as te, NLP, Text_input as ti

def nav():
    option = st.sidebar.selectbox("Navigation", ['About Application', 'Dataset input analysis', 'Raw Text input analysis'])
    if 'Dataset input analysis' in option:
        uploader()
    if 'About Application' in option:
        about()
    if 'Raw Text input analysis' in option:
        ti.text_input()
    
        
        
def about():
    col1, col2, col3 = st.columns((1,3,1))
    with col2:
        st.title("Text Analytics Application 💬")
        st.write('---')
        st.header("What is Text Analytics?")
        st.warning('''
        Text mining, also referred to as text data mining, similar to text analytics, is the process of deriving high-quality information from text. 
        It involves "the discovery by computer of new, previously unknown information, by automatically extracting information from different written resources."
        ''')
        st.header("About Application")
        st.warning('''
        This application is a type of software that aids in the creation of organized text data from unstructured text, as well as the discovery of trends, insights, and trends.
        It is capable of automating EDA process and incorporating machine learning models into the input data to provide seamless workflow to the end-users.
        ''')
        # st.header("Tools and Techniques")
        st.subheader("Accepeted Data Input")
        st.markdown('''
        > 
            - CSV File Input
            - Raw Text Input
        
        ''')
        st.subheader("Machine Learning Models")
        st.markdown('''
        > 
            - Logistic Regression model
            - Multinomial Naive Bayes Classifier
            - Decision Tree Classifier
            - SGD Classifier (Text Input)
        ''')
        st.subheader("Want to learn more?")
        st.info("**Select an option from the navigation bar on the left** to kick start your analysis!")

def uploader():
    uploaded_file = st.sidebar.file_uploader(
        "Upload a new input dataset",
        type = [
            'csv',
            'xlsx'
        ]
    )
    if uploaded_file is not None:
        df = da.load_dataset(uploaded_file)
        st.sidebar.write('---')
        text_sel(df)
    else:
        st.sidebar.info("Awaiting for CSV file to be uploaded.")
        if st.sidebar.checkbox('Press to use Example Dataset'):
            df = da.load_dataset('Dataset/twitter_racism_parsed_dataset.csv')
            st.sidebar.write('---')
            text_sel(df)
            

def text_sel(df):
    # with st.expander("Select a text"):
    st.subheader("Select a text")
    menu = ['Please Select']
    option = st.selectbox("Choose one column",menu + list(df.columns))
    st.caption("Select one column with your texts.")
    check = st.checkbox("Preview dataset")
    if check:
        st.title("Uploaded dataset")
        util.AgGridTable(df)
    if 'Please Select' not in option:
        df[option] = df[option].astype("str").astype("string")
        new_df = df[option]
        dashboard(new_df, option)

def dashboard(new_df, option):
    df = pd.DataFrame(new_df)
    df = df.rename(columns={df.columns[0]: 'Text'})
    st.title("Overview")
    total_observation = len(df.index)
    total_duplicated = int(df.duplicated().sum())
    unique_count = len(pd.unique(df['Text']))
    df['totalwords'] = df['Text'].str.split().str.len()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Column", option)
        st.metric("Total Observation", total_observation)
        st.metric("Total Duplicated records", total_duplicated)
        st.metric("Total unique values", unique_count)
    with col2:
        fig = px.histogram(df, x="totalwords", nbins=80)
        fig.update_layout(title_text='Text Length per row',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width = 700)
        st.plotly_chart(fig)
    st.sidebar.write("**Build Analysis**")
    option = st.sidebar.multiselect("Select one or more options to perform the analysis", ['Text Analytics', 'NLP Overview'])
    if 'Text Analytics' in option:
        te.text_extraction(new_df)
    if 'NLP Overview' in option:
        NLP.text_cleaning(new_df)
    

def download_but(df, file_name):
    st.download_button(
        label='Download CSV',
        data=df.to_csv(),
        file_name=file_name)




