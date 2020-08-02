import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()
from helpers import get_database_url
import os
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.decomposition import TruncatedSVD
from wordcounter import WordCounter
import sys

def load_data(database_name):
    url = get_database_url()
    engine = create_engine(url)
    df = pd.read_sql_table(database_name, engine)
    X = df["message"]
    Y = df[df.columns[3:]]
    Y = Y.astype(int)
    categories = list(Y.columns)
    return X, Y, categories

def tokenize(series):
    
    stop_words = stopwords.words("english")
    
    tokenizer = RegexpTokenizer("\w+|\d+")
    
    tokens = []
    
    for row in series:
        clean = tokenizer.tokenize(row.lower())
        tokens.append(clean)
        
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        #tok = list(set(tok) - set(stop_words))
        clean_tok = lemmatizer.lemmatize(str(tok)).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pipeline = Pipeline(
        [
            ("features", FeatureUnion([
                ("text_pipeline", Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                ])),
                ("word_count", Pipeline([
                    ('text_len', WordCounter()),
                ])),
            ])),
            ("clf", MultiOutputClassifier(RandomForestClassifier()))
        ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = Y.columns)
    for column in Y_test.columns:
        precision, recall, fscore, support = score(Y_test[column], Y_pred_df[column], average="weighted")
        print(column, precision, recall, fscore)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()