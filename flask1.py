from flask import Flask, request, jsonify, render_template
import joblib
import traceback
import pandas as pd
import numpy as np

lr = joblib.load("model.pkl") # Load "model.pkl"

# Your API definition
app = Flask(__name__)
@app.route('/')
def index():
  return render_template('index.html')
  
@app.route('/predict', methods=['POST'])
def predict():
    #if lr:
        
    int_features = [int(x) for x in request.form.values()]
    value = int_features[0]
            #json_ = request.json
            #print(json_)
            #value = int(json_['value'])
    prediction = list(lr.predict(value,value))

            #return jsonify({'Prediction for SR at Peak Demand (MW)': str(prediction)})
    return render_template('index.html', prediction_text='Next SR at Peak Demand (MW) should be {}MW'.format(prediction[0]))
        
       

        
    #else:
    #    print ('Train the model first')
    #    return ('No model here to use')

if __name__ == '__main__':
    port = 8888
    app.run(port=port)
    
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model columns loaded')

    
