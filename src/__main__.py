import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from models import run_all_models, cross_validation
from sampling import over_sampling
from import_and_split import read_data, split_and_vectorise


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
    




