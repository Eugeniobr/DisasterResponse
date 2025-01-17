import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    new_df = df.drop(columns=['id', 'message', 'original' ,'genre'])
    category_names = new_df.sum().sort_values(ascending=False)[:5].index
    category_values = new_df.sum().sort_values(ascending=False)[:5].values
    
    natural_disasters = ['floods', 'fire', 'storm', 'fire', 'earthquake']

    missing_people = []

    for type_of_disaster in natural_disasters:
        missing_people.append(new_df[new_df['missing_people'] == 1][type_of_disaster].sum())
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    
        {   
            'data':[
                
                Bar(
                    x = category_names,
                    y = category_values
                ) 
            ],
        
            'layout':{
                'title': 'Absolute Frequency of the 5 most frequent categories',
        
                    'yaxis':{
                        'title':'Count'
                    },
                    
                    'xaxis':{
                        'title': 'Category'
                    }   
            }
        },
        
        {   
            'data':[
                
                Bar(
                    x = natural_disasters,
                    y = missing_people
                ) 
            ],
        
            'layout':{
                'title': 'Missing people according to natural disaster',
        
                    'yaxis':{
                        'title':'Count'
                    },
                    
                    'xaxis':{
                        'title': 'Natural disaster'
                    }   
            }              
            
        }       
    ]
    
    new_df = df.drop(columns=['id', 'message', 'original' ,'genre'])
    category_names = new_df.sum().sort_values(ascending=False)[:5].index
    category_values = new_df.sum().sort_values(ascending=False)[:5].values
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()