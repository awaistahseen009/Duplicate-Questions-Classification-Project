import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize,sent_tokenize
import pandas as pd
import re
from nltk.stem import PorterStemmer
import distance
from fuzzywuzzy import fuzz
import string
import pickle
import streamlit as st
stemmer=PorterStemmer()
st.set_page_config(
    page_title="Duplicate Questions Classification",
    page_icon="favicon.ico",
)
# Short words dictionary because many people use short words in text
short_words = {
    "u":"you",
    "4u":"for you",
    "4":"for",
    "2":"to",
    "2u":"to you",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.replace(',000,000,000 ', 'b ')
    text = text.replace(',000,000 ', 'm ')
    text = text.replace(',000 ', 'k ')
    text = text.replace('%', ' percent')
    text = text.replace('$', ' dollar ')
    text = text.replace('₹', ' rupee ')
    text = text.replace('€', ' euro ')
    text = text.replace('@', ' at ')
    text = text.replace('[math]', '')
    text = re.sub(r'([0-9]+)000000000', r'\1b', text)
    text = re.sub(r'([0-9]+)000000', r'\1m', text)
    text = re.sub(r'([0-9]+)000', r'\1k', text)

    tokens = word_tokenize(text)
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [short_words.get(token, token) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]

    text = ' '.join(tokens)
    text = text.replace("'ve", " have")
    text = text.replace("n't", " not")
    text = text.replace("'re", " are")
    text = text.replace("'ll", " will")

    return text

def init_preprocess(q1, q2):
    df = pd.DataFrame({'question1': [q1], 'question2': [q2]})  # Pass lists as values
    q1_pre = preprocess_text(q1)
    q2_pre = preprocess_text(q2)
    df['clean_question1'] = q1_pre
    df['clean_question2'] = q2_pre
    df['n_words_q1'] = df['clean_question1'].apply(lambda x: len(word_tokenize(x)))
    df['n_sentences_q1'] = df['clean_question1'].apply(lambda x: len(sent_tokenize(x)))
    df['length_sentence_q1'] = df['clean_question1'].apply(len)
    df['n_words_q2'] = df['clean_question2'].apply(lambda x: len(word_tokenize(x)))
    df['n_sentences_q2'] = df['clean_question2'].apply(lambda x: len(sent_tokenize(x)))
    df['length_sentence_q2'] = df['clean_question2'].apply(len)
    return df

def basic_features(row):
    w1 = set(map(lambda word: word.lower().strip(), row['clean_question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['clean_question2'].split(" ")))
    common_words = len(w1 & w2)
    row['common_words'] = common_words

    w1 = set(map(lambda word: word.lower().strip(), row['clean_question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['clean_question2'].split(" ")))

    total_words = len(w1) + len(w2)
    row['total_words'] = total_words

    if total_words != 0:
        row['word_share'] = round(row['common_words'] / row['total_words'], 2)
    else:
        row['word_share'] = 0.0
    return row

def token_features(row):
    q1_tokens = row['clean_question1'].split(" ")
    q2_tokens = row['clean_question2'].split(" ")
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return row
    row['abs_length_tokens'] = abs(len(q1_tokens) - len(q2_tokens))
    row['mean_length_tokens'] = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(row['clean_question1'], row['clean_question2']))
    if len(strs) == 0:
        row['substring_dis'] = 0
    else:
        row['substring_dis'] = len(strs[0]) / (min(len(row['clean_question1']), len(row['clean_question2'])) + 1)

    return row

def fuzzy_features(row):
    row['fuzz_ratio'] = fuzz.QRatio(row['clean_question1'], row['clean_question2'])
    row['fuzz_partial_ratio'] = fuzz.partial_ratio(row['clean_question1'], row['clean_question2'])
    row['token_sort_ratio'] = fuzz.token_sort_ratio(row['clean_question1'], row['clean_question2'])
    row['token_set_ratio'] = fuzz.token_set_ratio(row['clean_question1'], row['clean_question2'])
    return row

def get_word_embeddings(text, model):
    embeddings = []
    for word in text:
        if word in model.wv:
            embeddings.append(model.wv[word])
    return embeddings

def get_mean_embeddings(text, model, remove_stopwords=True):
    tokens = word_tokenize(text.lower())
    embeddings = get_word_embeddings(tokens, model)
    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
    else:
        mean_embedding = np.zeros(model.vector_size)
    return mean_embedding

def pipeline(q1,q2):
    if q1 and q2:
        df = init_preprocess(q1, q2)
        df=df.apply(basic_features,axis=1)
        df=df.apply(token_features,axis=1)
        df=df.apply(fuzzy_features,axis=1)
        word2vec=pickle.load(open("word2vec_duplicate.pkl","rb"))
        rfc=pickle.load(open("rfc_duplicate.pkl",'rb'))
        df['embeddings_q1'] = df['clean_question1'].apply(lambda x: get_mean_embeddings(x, word2vec))
        df['embeddings_q2'] = df['clean_question2'].apply(lambda x: get_mean_embeddings(x, word2vec))
        emb_question1=np.vstack(df['embeddings_q1'])
        emb_question2=np.vstack(df['embeddings_q2'])
        rem_data=df.drop(['question1','question2','clean_question1','clean_question2','embeddings_q1','embeddings_q2'],axis=1)
        final_dataset=np.concatenate((emb_question1,emb_question2,rem_data),axis=1)
        prediction=rfc.predict(final_dataset)[0]
        if prediction==0:
            st.subheader("Not Duplicate")
        else:
            st.subheader("Question is duplicate")

    else: 
        st.subheader("Enter both questions please")

st.title('Duplicate Questions Classification')

q1=st.text_area('Enter the question 1')
q2=st.text_area('Enter the question 2')

if st.button('Predict'):
    pipeline(q1=q1,q2=q2)
