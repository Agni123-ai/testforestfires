import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request,jsonify

application = Flask(__name__)

## import ridge regressor and standard sacler pickle
ridge_model = pickle.load(open('models/Ridge.pkl', 'rb'))
Standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@application.route('/')
def index():
    return render_template("index.html")
  
@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
  if request.method == "POST":
    temperature = float(request.form.get('Temperature'))
    rh = float(request.form.get('RH'))
    ws = float(request.form.get('WS'))
    rain = float(request.form.get('Rain'))
    ffmc = float(request.form.get('FFMC'))
    dmc = float(request.form.get('DMC'))
    isi = float(request.form.get('ISI'))
    classes = float(request.form.get('Classes'))
    region = float(request.form.get('Region'))  
    
    new_data=Standard_scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
    pred= ridge_model.predict(new_data)
    return render_template("home.html", result=pred[0])
  else:
    return render_template("home.html")  

if __name__ == '__main__':
    application.run(debug=True)