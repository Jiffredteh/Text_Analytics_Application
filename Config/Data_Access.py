import streamlit as st
import pandas as pd

###########################################
#              File Loader                #
###########################################
@st.cache_data
def load_dataset(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
    except ValueError:
        df = pd.read_csv(uploaded_file)
    return df
