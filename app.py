#Importing necessary packages for the flask app
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import pickle

#Loading the necessary model
with open('best_model.pk1','rb') as f:
    model=pickle.load(f)
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    data=request.form.to_dict()
    #Processing the inputs
    input_data=pd.DataFrame([data])
    prediction=model.predict(input_data)[0]
    return jsonify({'prediction':f"${prediction:,.2f}"})
if __name__=='__main__':
    app.run(debug=True)