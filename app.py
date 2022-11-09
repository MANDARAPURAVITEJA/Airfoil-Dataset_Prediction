import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app=Flask(__name__)

#loading a pickle file
model=pickle.load(open('model.pkl','rb'))

#Home page for enterring data
@app.route('/')
def Home():
    return render_template("Home.html")

#creating the api
@app.route('/predict_api',methods=['POST'])
def predict_api(): #for predicting single output from model

    #for getting json data from postman
    data=request.json['data']
    print(data)
    new_data=[list(data.values())] #converting single data into 2-d array
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)[0]
    print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))


if __name__=="__main__":
    app.run(debug=True)