# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:33:15 2022

@author: Abdullah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def marks_prediction(hrs):
    data = pd.read_csv("student_scores.csv")
    
    X = data.iloc[:,:-1].values
    y = data.iloc[:,1].values
    
    model = LinearRegression()
    model.fit(X, y)

    X_test = np.array(hrs)
    X_test = X_test.reshape((1, -1))
    
    return model.predict(X_test)

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello():
    marks_pred = 0
    if request.method == "POST":
        hrs = request.form["hours"]
        marks_pred = marks_prediction(hrs)
        print(marks_pred)
    return render_template("index.html", mp = marks_pred )

if __name__ == "__main__":
    app.run(debug=True)
    
