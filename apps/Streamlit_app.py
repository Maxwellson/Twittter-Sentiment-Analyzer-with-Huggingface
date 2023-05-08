import streamlit as st 
import numpy as np
import transformers
import torch
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import select
from PIL import Image



# Load the dataset
df = pd.read_csv('Eval_subset.csv')
st.set_page_config(
    page_title="Twitter Sentiment Analyzer", page_icon="📊", layout="wide"
)

#Image Description
st.image = Image.open("sentiment inage 2.jpg")


@st.cache_resource
def get_model():
    model = transformers.AutoModelForSequenceClassification.from_pretrained("Kwasiasomani/Finetuned-Distilbert-base-model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Kwasiasomani/Finetuned-Distilbert-base-model")
    return tokenizer,model

tokenizer, model = get_model()

st.header("Covid19 Twitter Sentimental Analysis")
st.text("This app uses Distilbert-base-uncased for the analysis.")
add_text_sidebar = st.sidebar.title("Menu")
with st.sidebar:
    st.title("Twitter Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This app performs sentiment analysis on the latest tweets based on 
            the entered search term. Since the app can only predict positive or 
            negative, and neutral sentiment, it is more suitable towards analyzing the 
            sentiment of brand, product, service, company, or person. 
            Only English tweets are supported.
        </div>
        """,
        unsafe_allow_html=True,
    )

#Graphical representation
st.sidebar.markdown("### Number of tweets by sentiment")
sentiment_count = df["safe_text"].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, "Tweets":sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == "Bar plot":
        fig = px.bar(sentiment_count, x="Sentiment", y="Tweets", color="Tweets", height=50)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values="Tweets", names="Sentiment")
        st.plotly_chart(fig)





# Side view description 
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio("Sentiment", ("positive", "neutral", "negative"))
st.sidebar.markdown(df["safe_text"])


st.markdown("[Github link](https://github.com/Norkplim22)")
st.markdown("[Medium link](https://medium.com/@cnorkplim)")
st.markdown("Created by Foster,Kwasi,Linda,Stella,Joshua and Bright")
user_input = st.text_area("Enter text to predict")
button = st.button("predict")


# Define Helper Function
label = {0: 'Negative', 1:'Neutral', 2:'Positive'}


      
# Prediction
if user_input and button:
    test_input = tokenizer([user_input],return_tensors='pt')
    st.slider("Number of tweets", min_value=100, max_value=2000, key="user_input")

    # Test output
    output = model(**test_input)
    st.write('Logits:',output.logits)
    predicted_class = np.argmax(output.logits.detach().numpy())
    st.write('prediction:',label[predicted_class])