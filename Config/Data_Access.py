import streamlit as st
import pandas as pd

###########################################
#              File Loader                #
###########################################
@st.cache(allow_output_mutation=True)
def load_dataset(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
    except ValueError:
        df = pd.read_csv(uploaded_file)
    return df