
import pandas as pd
import numpy as np

# Data splitting and modelling
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def split_and_vectorise():
    df = pd.read_csv("clean_stem_token_data.csv")
    #Making list into string so I can vectorise
    df["test"] = df["stem_review"].apply(" ".join)
    print(df["test"])
    #Using tfidf vect
    tfidf_vect = TfidfVectorizer(max_features=5000)

    #Splitting X and Y data
    X_train = df["test"]
    print(X_train)
    y = df["rating"]
    print(y)

    #Tfidf vect variable
    X = tfidf_vect.fit_transform(X_train)
    print(X)

    #Splitting dataset into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #Return split data
    return X_train, X_test, y_train, y_test



def logstic_regression(x_train, x_test, y_train, y_test):
    #Create logstic regression model
    log_model = LogisticRegression()
    #Fit model
    log_model = log_model.fit(X=x_train, y=y_train)
    #Predict
    y_pred = log_model.predict(x_test)
    #Overall accuracy of model
    accuracy = accuracy_score(y_test, y_pred)
    print("Overall Logistic regression accuracy is: {}%".format(accuracy))

   

def bayes(x_train, x_test, y_train, y_test):
    #Bayes Model
    #Had to use toarray because X_train was a sparse matrix
    classifier = GaussianNB()
    #Fitting
    classifier = classifier.fit(x_train.toarray(), y_train)
    #Prediciton
    y_pred_NB = classifier.predict(x_test.toarray())
    #Overall accuracy
    accuracy = accuracy_score(y_test, y_pred_NB)
    print("overall Bayes accuracy is : {}%".format(accuracy))
    return accuracy

def random_forest(x_train, x_test, y_train, y_test):
    rf_model = RandomForestClassifier()
    #Fitting 
    rf_model = rf_model.fit(x_train.to_array(), y_train)
    #Predict
    y_predict_rf = rf_model.predict(x_test.toarray())
    #Overall accuracy
    accuracy = accuracy_score(y_test, y_predict_rf)
    print("Overall RF accuracy is {}%.".format(accuracy))
    return accuracy


# # Over/Undersampling: Splitting data, oversampling, vectorising then returning x_train, x_test, y_train and y_test
def over_sampling():
    #Reading data
    df = pd.read_csv("clean_stem_token_data.csv")
    #Splitting train and test before X and y so model doesn't test on seen data
    train_df, test_df = train_test_split(df, test_size = 0.2)
    #Oversampling ratings 1-3
    rating_3 = pd.concat([train_df[train_df["rating"] == 3.0]]*3, ignore_index = True)
    rating_2 = pd.concat([train_df[train_df["rating"] == 2.0]]*4, ignore_index = True)
    rating_1 = pd.concat([train_df[train_df["rating"] == 1.0]]*3, ignore_index = True)
    #COncatting oversampling to train data
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
    X_final_test = tfidf_vect.fit_transform(X_resample_test) #Final X test data to be returned

    #Returning resampled data
    return X_train, X_final_test, y_train, y_resample_test

#Putting cross val up here because the model functions will use it
def cross_validation(model, x, y):

    scores = cross_val_score(model, x, y, scoring= "accuracy", cv = 10)
    print("Cross Validation accuracy is {}%".format(scores))
    return scores

if __name__ == "__main__":
    split_data = split_and_vectorise()
    #Main train and test
    x_train, x_test, y_train, y_test = split_data[0], split_data[1], split_data[2], split_data[3]
    print( x_train, x_test, y_train, y_test)
    """
    logstic_regression(x_train, x_test, y_train, y_test)
    bayes(x_train, x_test, y_train, y_test)
    random_forest(x_train, x_test, y_train, y_test)
    #Resampling train and test
    resample_data = over_sampling()
    #Main train and test
    x_train_re, x_test_re, y_train_re, y_test_re = resample_data[0], resample_data[1], resample_data[2], resample_data[3]
    logstic_regression(x_train_re, x_test_re, y_train_re, y_test_re)
    bayes(x_train_re, x_test_re, y_train_re, y_test_re)
    random_forest(x_train_re, x_test_re, y_train_re, y_test_re)
    
"""






