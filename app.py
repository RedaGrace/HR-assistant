from flask import *
import pandas as pd
import json, time
import pickle
import re
import os
from flask_cors import CORS

filename = 'hr_rf2.pickle'
def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model
  
# Read in pickle
rf2 = read_pickle(path, 'hr_rf2')

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET'])
def home_page():
    data_set = {'Page': 'Home', 'Message': "Let's get started and send me your inputs", 'Timestamp': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/predict/', methods=['GET'])
def request_page():
    inputs = request.args.get('inputs')                              # /predict/?inputs=inputs
    inputs= [float(n) for n in re.findall('[-+]?(?:\d*\.*\d+)', inputs)]
    

    output = rf2.predict([inputs])[0]
    proba = rf2.predict_proba([inputs])[0].max()

    data_set = {'prediction': output,
                'probability': proba
                
                }
                     
    json_dump = json.dumps(data_set)
    return json_dump

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)