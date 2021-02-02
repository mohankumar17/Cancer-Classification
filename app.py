import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('CancerClassifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    fl_features = [float(x) for x in request.form.values()]
    final_features = [np.array(fl_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    output = prediction[0]
    '''if pred == 0:
        output = 'Benign'
    else:
        output = 'Malignant'
    '''

    return render_template('index.html', prediction_text='Cancer Type is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)