import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('new_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #exp = int(request.form['experience'])
    #two = int(request.form['test_score'])
    #three = int(request.form['interview_score'])
    
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #final_features = [[one, two, three]]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Outlet_sales {}'.format(output))
    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(final_features))


if __name__ == "__main__":
    app.run(debug=True)
    