from flask import Flask, render_template,request
import numpy as np
import joblib

app = Flask(__name__)
model=joblib.load('breast_cancer_detector.pkl')
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    ##output = round(prediction[0]/1E9, 2)
    if prediction == np.array([1]):
        return render_template('index.html', 
                            prediction_text='Patient has Cancer')
    if prediction == np.array([0]):
        return render_template('index.html', 
                            prediction_text='Patient does not have Cancer')

if __name__ == "__main__":
    app.run()
