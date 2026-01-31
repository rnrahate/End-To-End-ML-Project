from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score_str = request.form.get('reading_score')
        writing_score_str = request.form.get('writing_score')
        
        if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score_str, writing_score_str]):
            return render_template('home.html', error="All fields are required.")
        
        assert gender is not None
        assert race_ethnicity is not None
        assert parental_level_of_education is not None
        assert lunch is not None
        assert test_preparation_course is not None
        assert reading_score_str is not None
        assert writing_score_str is not None
        
        try:
            reading_score = float(reading_score_str)
            writing_score = float(writing_score_str)
        except ValueError:
            return render_template('home.html', error="Invalid score values.")
        
        data=CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])