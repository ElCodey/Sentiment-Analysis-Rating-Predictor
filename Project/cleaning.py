import pandas as pd
import nltk
import string 
import re
# Preprocessing/analysis
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#Downloads
nltk.download("stopwords")
nltk.download("wordnet")

def clean(data):
    df = data
    #Make column lower
    df["lower"] = df["text_review"].str.lower()
    #Remove punc
    df["no_punc"] = df["lower"].str.replace(r"[^\w\s]", "")
    #Removing numbers
    df["remove_numbers"] = df["no_punc"].str.replace(r"\d+", "")
    #Tokenise
    df["tokenise"] = [re.split(r"\W+", word) for word in df["remove_numbers"]]
    #Drop extra columns before returning
    df.drop(["lower", "no_punc", "remove_numbers",], axis=1, inplace=True)

    return df

def remove_stopwords(tokenised_text):
    stop_words = stopwords.words("english")
    cleaned_text = [word for word in tokenised_text if word not in stop_words]
    return cleaned_text

def stem(text):
    #Stemming the words
    ps = PorterStemmer()
    stem_text = [ps.stem(word) for word in text]

    return stem_text

def final_clean(data):
    df = data
    #Run stopwords function
    df["no_stopwords"] = df["tokenise"].apply(lambda x: remove_stopwords(x))
    #Run stemming function
    df["stem_review"] = df["no_stopwords"].apply(lambda x: stem(x))

    return df
