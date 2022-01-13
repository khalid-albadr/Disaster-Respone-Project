# Disaster Response Pipeline Project

### Dependencies
1.Python 3.5+ (I used Python 3.10)
2.Machine Learning Libraries: NumPy,Pandas, Sciki-Learn
3.Natural Language Process Libraries: NLTK
4.SQLlite Database Libraqries: SQLalchemy
5.Web App and Data Visualization: Flask, Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
