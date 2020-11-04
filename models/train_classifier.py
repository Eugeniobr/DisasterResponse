import pandas as pd
import nltk
from nltk import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re
import sys
import pickle
nltk.download(['punkt', 'stopwords'])





def load_data(database_filepath):
    """
	INPUT: database_filepath
    
	OUTPUT: X is the dataframe that contains messages
		Y is the dataframe that contains the labels
		category_names contains the name of each category
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("select * from DisasterResponse", con=engine)
    X = df['message'] 
    Y = df.drop(columns=['message','id', 'original', 'genre']) 
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """
    
        INPUT: text is the message to be processed
       
        OUTPUT: tokens is a list of tokens obtained after processing the message
    
    """    
    lemmatizer = WordNetLemmatizer()
    lower_case_text = text.lower()
    loowe_case_text = re.sub(r'[^\w\s]', '', lower_case_text)
    tokens = word_tokenize(lower_case_text)
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return lemmatized_tokens


def build_model():
    
    model = Pipeline([
    
    ('vect', CountVectorizer()),
    ('Tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    
	])

    return model

def tune_model(model):
    """
        INPUT: model- model with hyperparameters to be tuned 
        
        OUTPUT: tuned_model - model with hyperparameters tuned 
    """
    parameters = {
    'clf__estimator__weights': ['uniform', 'distance']
    }

    tuned_model = GridSearchCV(model, param_grid=parameters)
    
    return tuned_model

def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    cont = 0
    for column in Y_test.columns:
	    #accuracy
        print("Accuracy:")
        print("{:.2f}".format(accuracy_score(Y_test[column].values, Y_pred[:,cont])))
        #precision, recall and f1-score 	
        print(classification_report(Y_test[column].values, Y_pred[:,cont]))
        cont = cont + 1  



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        tuned_model = tune_model(model) 
        tuned_model.fit(X_train, Y_train)
        
        
        print('Evaluating model...')
        evaluate_model(tuned_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(tuned_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()