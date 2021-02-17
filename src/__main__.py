import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from models import run_all_models, cross_validation
from sampling import over_sampling

def read_data():
    df = pd.read_csv("clean_stem_token_data.csv")
    return df


def split_and_vectorise(df):
    
    #Making list into string so I can vectorise
    df["test"] = df["stem_review"].astype(str)
    
    #Using tfidf vect
    tfidf_vect = TfidfVectorizer(max_features=5000)

    #Splitting X and Y data      
    y = df["rating"]
    
    #Tfidf vect variable
    X = tfidf_vect.fit_transform(df["test"])
   
    #Splitting dataset into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    #Return split data
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    #Import data and vectorise
    df = read_data()
    vectorise = split_and_vectorise(df)
    #Run all models
    all_models = run_all_models(vectorise[0], vectorise[1], vectorise[2], vectorise[3])
    #Oversampling
    sample_df = over_sampling(df)    
    #Run oversampling
    resample_all_models = run_all_models(sample_df[0], sample_df[1], sample_df[2], sample_df[3])   
    #Cross validation
    cv = cross_validation(df["test"], df["rating"])
    




