import pandas as pd
import numpy as np
from scipy.special import softmax
import gradio as gr
import torch
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from torch import nn



# Define the model path where the pre-trained model is saved on the Hugging Face model hub
model_path = "slickdata/finetuned-Sentiment-classfication-ROBERTA-model"

# Initialize the tokenizer for the pre-trained model
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Load the configuration for the pre-trained model
config = AutoConfig.from_pretrained(model_path)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define a function to preprocess the text data
def preprocess(text):
    new_text = []
    # Replace user mentions with '@user'
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        # Replace links with 'http'
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    # Join the preprocessed text
    return " ".join(new_text)

# Define a function to perform sentiment analysis on the input text
def sentiment_analysis(text):
    # Preprocess the input text
    text = preprocess(text)

    # Tokenize the input text using the pre-trained tokenizer
    encoded_input = tokenizer(text, return_tensors='pt')

    # Feed the tokenized input to the pre-trained model and obtain output
    output = model(**encoded_input)

    scores_ = softmax(output.logits[0].detach().numpy())

    # Format the output dictionary with the predicted scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }

    # Get the label with the highest score
    max_score_label = max(scores, key=scores.get)

    # Return the label with the highest score
    return max_score_label

# Define a Gradio interface to interact with the model
demo = gr.Interface(
    fn=sentiment_analysis, # Function to perform sentiment analysis
    inputs=gr.Textbox(placeholder="Write your tweet here..."), # Text input field
    outputs="label", # Output type (here, we only display the label with the highest score)
    interpretation="default", # Interpretation mode
    examples=[["This is wonderful!"]]) # Example input(s) to display on the interface

# Launch the Gradio interface
demo.launch(share=True, debug=True)