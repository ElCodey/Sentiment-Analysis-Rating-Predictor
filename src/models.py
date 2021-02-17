
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




def logstic_regression(x_train, x_test, y_train, y_test):
    #Create logstic regression model
    log_model = LogisticRegression()
    #Fit model
    log_model = log_model.fit(X=x_train, y=y_train)
    #Predict
    y_pred = log_model.predict(x_test)
    #Overall accuracy of model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

   

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
    return accuracy

def random_forest(x_train, x_test, y_train, y_test):
    rf_model = RandomForestClassifier()
    #Fitting 
    rf_model = rf_model.fit(x_train.toarray(), y_train)
    #Predict
    y_predict_rf = rf_model.predict(x_test.toarray())
    #Overall accuracy
    accuracy = accuracy_score(y_test, y_predict_rf)
    return accuracy

#Function to run all 3 models at the same time 
def run_all_models(x_train, x_test, y_train, y_test):
    #Log model
    log_reg_results = logstic_regression(x_train, x_test, y_train, y_test)
    #Bayes
    bayes_results = bayes(x_train, x_test, y_train, y_test)
    #Random Forest
    random_forest_results = random_forest(x_train, x_test, y_train, y_test)

    return log_reg_results, bayes_results, random_forest_results


#Putting cross val up here because the model functions will use it
def cross_validation(X, y):
    x = TfidfVectorizer(max_features=5000).fit_transform(X)
    #Log regression
    log_model = LogisticRegression()
    log_score = cross_val_score(log_model, x, y, scoring= "accuracy", cv = 10)
    #RF model
    rf_model = RandomForestClassifier()
    rf_score = cross_val_score(rf_model, x.toarray(), y, scoring= "accuracy", cv = 10)
    #NB
    nb_model = GaussianNB()
    nb_score = cross_val_score(nb_model, x.toarray(), y, scoring= "accuracy", cv = 10)
    
    return log_score.mean(), rf_score.mean(), nb_score.mean()
    
