
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# # Over/Undersampling: Splitting data, oversampling, vectorising then returning x_train, x_test, y_train and y_test
def over_sampling(df):
    
    #Splitting train and test before X and y so model doesn't test on seen data
    train_df, test_df = train_test_split(df, test_size = 0.2)
    #Oversampling ratings 1-3
    rating_3 = pd.concat([train_df[train_df["rating"] == 3.0]]*3, ignore_index = True)
    rating_2 = pd.concat([train_df[train_df["rating"] == 2.0]]*4, ignore_index = True)
    rating_1 = pd.concat([train_df[train_df["rating"] == 1.0]]*3, ignore_index = True)
    #Concatting oversampling to train data
    train_df = pd.concat([train_df, rating_3, rating_2, rating_1])
    #Splitting X and y for train
    X_resample = train_df["test"]
    y_train = train_df["rating"]  #Final y train data to be returned

    tfidf_vect = TfidfVectorizer(max_features=5000)
    #Tfidf vect variable
    X_train = tfidf_vect.fit_transform(X_resample) #Final x train data to be returned


    #Test dat
    X_resample_test = test_df["test"]
    y_resample_test = test_df["rating"] #Final Y test data to be returned

    
    #Vectorising X test
    X_final_test = tfidf_vect.transform(X_resample_test) #Final X test data to be returned

    #Returning resampled data
    return X_train, X_final_test, y_train, y_resample_test

