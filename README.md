# Disaster Response Pipeline Project
### Description
The dataset contains pre-labelled tweets and messages from real-life disasters provided by Fiqure Eight. The goal of the project is to build a model that categorize messages to disaster categories using best Data Engineering practices. This Project is part of Data Science Nanodegree Program by Udacity.

The Project is divided in the following Sections:

-ETL Pipeline to extract data, clean data and save them in a database file
-Machine Learning Pipeline to train a model able to classify text message in categories of disasters.
-Web App for visualizations.
### Dependencies
-Python 3.5+ (I used Python 3.10)

-Machine Learning Libraries: NumPy,Pandas, Sciki-Learn

-Natural Language Process Libraries: NLTK

-SQLlite Database Libraqries: SQLalchemy

-Web App and Data Visualization: Flask, Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
