import streamlit as st
import gdown
import pickle
import tensorflow as tf
import os
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
nltk.download('stopwords')
stop_words = stopwords.words('english')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from tensorflow.keras.preprocessing.sequence import pad_sequences

directory_path = 'C:/Users/PM/Desktop/'
pipreqs.generate_requirements_file(directory_path)

os.makedirs('content', exist_ok=True)
os.chdir('content')

gdown.download(id = '1qFViB4VmxIrC_atqunNzq9pqUaz0YMk0')
gdown.download(id = '1YGzqZhxig_5szsufoDXYJdRiAvxKtz1F')

# Load the tokenizer
with open('C:/Users/PM/.spyder-py3/content/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model('C:/Users/PM/.spyder-py3/content/model.h5')


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


def remove_stopwords(rev):

    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new


def badword_prediction(input_data):
     
    df = pd.DataFrame(columns = ['text'])
    new_row= {'text': input_data}
    df = df.append(new_row, ignore_index=True)
    
    df['text'] = [str(i).lower() for i in df['text']]
    df['text'] = df['text'].str.replace("[^a-zA-Z0-9]", " ") 
    
    df['text'] = df['text'].apply(lambda x: lemmatize_sentence(x))
    
    df['text'] = [remove_stopwords(r) for r in df['text']]
    
    texts = df['text'].to_list()

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    max_sequence_length = 15 # based on pad_seq length sugestion
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(padded_sequence)[0][0]
    threshold = 0.5
    predictions = ['it is a Bad word' if predictions >= threshold else 'it is not a Bad word' ]
       
    return(predictions)

def main():
    # titel
    st.title('Profanity Checker')
    
    # input
    text = st.text_input('Enter your text:')
    
    # prediction
    diagnosis = ""
    
    # button
    if st.button('Result'):
        diagnosis = badword_prediction(text)
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
