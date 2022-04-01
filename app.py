import streamlit as st
import pandas as pd
from Config import Global as glb

def layout():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="Text Analytics", page_icon="💬", layout="wide")
    with open('/Users/jiffred/Downloads/FYP/Text_Analytics_Dashboard/CSS/styling.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    hide_streamlit_style = """
                <style>
                .reportview-container .main .block-container{
                    {
                    padding-top: {
                        padding
                            }rem;
                    padding-right: {
                        padding
                            }rem;
                    padding-left: {
                        padding
                            }rem;
                    padding-bottom: {
                        padding
                            }rem;
                            }
                            }
                footer {
                    visibility: hidden;
                        }
                footer:after {
                        content:'developed by Jiffred Teh Yi Jia'; 
                        visibility: visible;
                        display: block;
                        position: relative;
                        #background-color: red;
                        padding: 5px;
                        top: 2px;
                        }
                </style>
                """
    st.markdown(
        hide_streamlit_style, 
        unsafe_allow_html=True
        )
    # st.markdown(
    # """
    # # 💬 Text Analytics Application
    # """ 
    # )
    # st.write('---')