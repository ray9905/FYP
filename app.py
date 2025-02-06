import streamlit as st
import pandas as pd 
import numpy as np

# Title
st.title('Welcome to NeuroAI')

# Header
st.write('Upload an EEG file to get started')

eeg_file = st.file_uploader('Upload EEG file', type=['csv'])

st.subheader('Results')
st.write('intepretation of data from llm will be shown here as a summary')