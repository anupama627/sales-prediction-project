from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
app = Flask(__name__)
data = pd.read_csv("sales_data.csv")
print(data.columns)  
data.columns = data.columns.str.strip()
X = data[['festival', 'weekend', 'salary_day', 'local_event', 'visitors', 'month']]
y = data['sales']
model = LinearRegression()
model.fit(X, y)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    festival = int(request.form['festival'])
    weekend = int(request.form['weekend'])
    salary = int(request.form['salary_day'])
    local = int(request.form['local_event'])
    visitors = int(request.form['visitors'])
    month = int(request.form['month'])

    features = np.array([[festival, weekend, salary, local, visitors, month]])
    prediction = model.predict(features)[0]

    return render_template('index.html', result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
    