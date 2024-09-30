import torch
import streamlit as st
from transformers import pipeline
from S3Manager import S3Manager


bucket_name = "mlops-with-kgp"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

s3_manager = S3Manager()

st.title("Machine Learning Model Deployment at the Server!!!")

button = st.button("Download Model")
if button:
    with st.spinner("Downloading... Please wait!"):
        s3_manager.download_s3_folder(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            local_path=local_path
        )
    

text = st.text_area(label="Enter Your Review", placeholder="Type your review here ...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
classifier = pipeline('text-classification', model='tinybert-sentiment-analysis', device=device)

if st.button("Predict"):
    with st.spinner("Predicting..."):
        output = classifier(text)
        st.write(output)