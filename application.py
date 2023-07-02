from email.mime import application
import numpy as np
import pandas as pd
import pickle
import json
from flask import Flask,render_template,url_for,jsonify,request

#create a Flask object
application = Flask(__name__)
#Loading the pickle files (model)
regmodel = pickle.load(open('Linear Regression Stock price.pkl','rb'))
scaler = pickle.load(open('Scaler Stock price.pkl','rb'))

@application.route('/',methods=['GET'])
def home():
    return render_template('Stock price prediction.html')

@application.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    result = regmodel.predict(new_data)
    print(result[0])
    return jsonify(result[0])

@application.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template('Stock price prediction.html',
                           prediction_text = "The Stock price prediction is {}".format(output))
application.run()

    


