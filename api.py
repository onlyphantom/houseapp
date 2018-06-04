import flask
from flask import Flask, render_template, request

import numpy as np
from scipy import misc

# Making Predictions using sk-learn
from sklearn.externals import joblib

# create an instance of the Flask class
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', label="No File")
        
        img = misc.imread(file)
        img = img[:,:,:3]
        img = img.reshape(1,-1)

        prediction = model.predict(img)

        label = str(np.squeeze(prediction))

        if label=='10':
            label='0'
        return render_template('index.html', label=label, file=file)

if __name__ == '__main__':
    model = joblib.load('models/rfClassifier.pkl')
    app.run(host='0.0.0.0', port=8181, debug=True)