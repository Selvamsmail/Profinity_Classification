
import streamlit as st
import gdown
import pickle
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

@st.cache_data
def collect():
  nltk.download('stopwords')
  stop_words = stopwords.words('english')
  nltk.download('omw-1.4')
  nltk.download('punkt')
  nltk.download('wordnet')
  nltk.download('averaged_perceptron_tagger')
  return

@st.cache_resource
def getdata():

  gdown.download(id = '1qFViB4VmxIrC_atqunNzq9pqUaz0YMk0')
  gdown.download(id = '1YGzqZhxig_5szsufoDXYJdRiAvxKtz1F')
  
  with open('/app/profinity_classification/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  model = load_model('/app/profinity_classification/model.h5')
  
  return tokenizer, model

def set_background(theme):

    encoded_image = 0
    
    page_bg_css = f'''
        <style>
            body {{
                background-image: url("data:image/jpeg;base64,{encoded_image}");
                background-size: cover;
            }}

            .stApp {{
                color: {theme['textColor']};
                background-color: {theme['backgroundColor']};
            }}
        </style>
    '''
    st.markdown(page_bg_css, unsafe_allow_html=True)

tokenizer, model = getdata()

lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  

    lemmatized_sentence = []
    
    for word, tag in nltk_tagged:
        if tag is None:

            lemmatized_sentence.append(word)
        else:
          try:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
          except:
            lemmatized_sentence.append(word)

    return " ".join(lemmatized_sentence) 


def badword_prediction(input_data):
     
    df = pd.DataFrame(columns = ['text'])
    new_row= {'text': input_data}
    new_row = pd.DataFrame(new_row, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    
    df['text'] = [str(i).lower() for i in df['text']]
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
    
    df['text'] = df['text'].apply(lambda x: lemmatize_sentence(x))
    
    texts = df['text'].to_list()

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    max_sequence_length = 15 # based on pad_seq length sugestion
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(padded_sequence)[0][0]
    threshold = 0.7
    predictions = np.exp(predictions) / (np.exp(predictions) + 1)
    confidence = predictions
    predictions = ('it is a Bad word with a confidence of '+str(predictions)+' %' if predictions >= threshold else 'it is not a Bad word' )
       
    return(predictions)

def main():
    a = collect()
    theme = {
        'base': 'dark',
        'backgroundColor': '#005C1D',
        'textColor': '#ffffff'
    }

    set_background(theme)

    st.write('Welcome to the, *Profanity Checker Application* :sunglasses:')
    st.title('Check your sentence')
    st.text('To get better results try typing words with atleast a length of 7')
    # input
    text = st.text_input('Enter your text:')
    
    # prediction
    ans = ""
    
    # button
    if st.button('Check'):
        ans = badword_prediction(text)
        
    st.success(ans)
    st.text('Another point is multiple checks do not change')
    st.text('the confidence level when the model is sure')
    st.write('soon an updated model will have regulations like *random_perturbations, data_rotation function, noise_injection* :muscle:')
    link = st.expander('*Project Link!* :point_down:', expanded=False)
    
    with link:
        st.markdown('Visit my project in this [link](https://github.com/Selvamsmail/Profinity_Classification)')


if __name__ == '__main__':
    main()
