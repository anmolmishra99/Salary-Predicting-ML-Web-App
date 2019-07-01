from flask import Flask, request, render_template, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
         #reading csv
        dataset = pd.read_csv('Salary_Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values

        #splititng dataset to test and training set

        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 42)

        #fitting Simple Linear Regression to the training set mean making machine learn by training set

        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train) 

        #predicting test set results

        y_pred = regressor.predict(X_test)


        if request.method == "POST":
                comment = request.form['comment']
                comment = float(comment)
                data = [comment]
                vect = [data]
                my_prediction = regressor.predict(vect)
                return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
