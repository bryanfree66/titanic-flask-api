
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from joblib import dump, load

app = Flask(__name__)
api = Api(app)

# load trained classifier
clf_path = 'models/v1/model.joblib'
with open(clf_path, 'rb') as f:
    model = load(f)


class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        id = posted_data['PassengerId']
        pclass = posted_data['Pclass']
        name = posted_data['Name']
        sex = posted_data['Sex']
        age = posted_data['Age']
        sibsp = posted_data['SibSp']
        parch = posted_data['Parch']
        ticket = posted_data['Ticket']
        fare = posted_data['Fare']
        cabin = posted_data['Cabin']
        embarked = posted_data['Embarked']

        prediction = model.predict(
            [[id,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]])[0]
        if prediction == 0:
            predicted_class = 'Did not Survive'
        elif prediction == 1:
            predicted_class = 'Survived'
        

        return jsonify({
            'Prediction': predicted_class
        })


api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
