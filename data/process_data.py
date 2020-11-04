import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
  """

    INPUT: messages_filepath is the file path to the message dataset
           categories_filepath is the filepath to the categories dataset     


    OUTPUT: df is the merged dataframe that combines the message and the categories dataset
  
  """
  messages_df = pd.read_csv(messages_filepath)
  categories_df = pd.read_csv(categories_filepath)  
  
  df = messages_df.merge(categories_df, on=['id'], left_index=True, right_index=True)

  return df

def clean_data(df):
  """

    INPUT: df is the merged dataframe

    OUTPUT: clean_df is the dataframe that results after cleaning process

  """
  # create a dataframe of the 36 individual category columns
  categories = df['categories'].str.split(";", expand=True)
  # select the first row of the categories dataframe
  row = categories.iloc[0]
  #remove the two last characters for each string in row
  category_colnames = row.apply(lambda x: x[:-3])
  # rename the columns of `categories`
  categories.columns = category_colnames
  #converts category values to numbers
  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int')

  # drop the original categories column from `df`
  df.drop(columns=['categories'], inplace=True)
  # concatenate the original dataframe with the new `categories` dataframe
  df = df.merge(categories, left_index=True, right_index=True)
  # drop duplicates
  df.drop_duplicates(inplace=True)

  return df


def save_data(df, database_filename):
  """
      INPUT: df is the cleaned dataframe
             database_filename is the name of the database which we will storing the cleaned data
  """
  engine = create_engine('sqlite:///'+ database_filename)
  df.to_sql('DisasterResponse', engine, index=False)


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