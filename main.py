from flask import Flask, request, render_template
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def start():
    return render_template("index.html")

@app.route("/predict", methods=['GET'])
def predict():
    if request.method == 'GET':
        file_name = request.args.get('chosenImg')
        return file_name
    else:
        return "here"
        
@app.route("/display", methods=['GET'])
def display():
    return os.listdir("templates")

if __name__ == '__main__':
    app.run(port=8080)

