# Disaster Response Pipeline Project

### Project Motivation:

This is the second project of the Udacity Data Scientist Nanodegree. In this project, I have implemented an ETL and a Machine Learning pipeline. The main
purpose of this project was to build a system that correctly classifies messages.

### Run:

In a terminal or command window, navigate to the project's root directory. Next, run the following command to clean and store the data into a database
using the ETL pipeline:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

In order to run the Machine Learning pipeline, type the following command:

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

To run the web app and see the visualizations, navigate to the app directory and type the command

`python run.py`

### File structure and description:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py   # python script that cleans and stores the dataset into a SQL database
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py # python script that trains and evaluates the machine learning model on the dataset


### Acknowledgments

Templates for the process_data.py and train_classifier.py scripts were provided by Udacity. The dataset was provided by Figure Eight. 
