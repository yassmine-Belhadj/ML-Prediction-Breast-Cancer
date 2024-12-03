import streamlit as st 
from PIL import Image
import pickle
import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

def predict(values, dic):
    # breast_cancer
    if len(values) == 22:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')



@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)


    if request.method == 'POST':
        try:
            img = Image.open(request.files['image'])
            img.save("uploads/image.jpg")
            img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
            os.path.isfile(img_path)
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img))
        except:
            message = "Please upload an image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)


if __name__ == '__main__':
    app.run(debug = True)