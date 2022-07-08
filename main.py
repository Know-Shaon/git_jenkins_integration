from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'welcome all'

@app.route('/predict', methods=['Get'])
def pred_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "The predicted value is: " + str(prediction)

@app.route('/predict_file', methods=['POST'])
def pred_note_file():
    df_test=pd.read_csv(request.files.get('file'))
    prediction=classifier.predict(df_test)
    return 'The prediction for the file is:'+ str(list(prediction))

if __name__ == '__main__':
    app.run(debug=True)