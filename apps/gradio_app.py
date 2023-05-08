from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import gradio as gr


tokenizer = AutoTokenizer.from_pretrained('lindaclara22/twitter-sentiment-analysis-roberta-model')
config = AutoConfig.from_pretrained('lindaclara22/twitter-sentiment-analysis-roberta-model')
model = AutoModelForSequenceClassification.from_pretrained('lindaclara22/twitter-sentiment-analysis-roberta-model')

# #Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    
    # Format output dict of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }
    
    return scores


demo = gr.Interface(
    img = gr.Image("sentiment image 1.png").style(height="24")
    Title=gr.Label("COVID SENTIMENT ANALYZER")
    description="This app performs sentiment analysis on tweets. Only English tweets are supported."
    fn=sentiment_analysis, 
    inputs=gr.Textbox(placeholder="Write your tweet here..."), 
    outputs="label", 
    interpretation="default",
    examples=[["This is Spectacular!"]])      


demo.launch()