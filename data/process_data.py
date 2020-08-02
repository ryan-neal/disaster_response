import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from helpers import get_database_url
load_dotenv()
import os
import sys

def load_data(message_filepath, category_filepath):
    messages = pd.read_csv(message_filepath, index_col=0)
    categories = pd.read_csv(category_filepath, index_col=0)
    df = messages.merge(categories, on="id")
    return df

def clean_data(df):
    get_name = lambda x:x.split('-')[0]
    get_number = lambda x:x.split('-')[1]
    categories = df["categories"].str.split(";", expand=True)
    category_columns = list(categories.iloc[0].apply(get_name))
    categories.columns = category_columns
    for column in categories:
        categories[column] = categories[column].apply(get_number)
        categories[column] = pd.to_numeric(categories[column])
    df.drop(["categories"], axis = 1, inplace=True)
    df = df.join(categories, on="id", how="outer")
    df.drop_duplicates(subset="message", inplace=True)
    return df

def save_data(df, database_name):
    url = get_database_url()
    engine = create_engine(url)
    df.to_sql(database_name, engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()