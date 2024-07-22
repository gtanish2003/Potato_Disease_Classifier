from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app=Flask(__name__)
model=load_model('my_model.h5')

# defining the classes in my project 
classes=['early_blight','late_blight','healthy']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file part"
    
    file=request.files['file']
    
    if file.filename==" ":
        return "No selected file"
    
    if file:
        filepath=os.path.join('uploads',file.filename)
        file.save(filepath)

        # preprocessing the image

        image = load_img(filepath, target_size=(150, 150)) 
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # if your model expects normalization

        # predict the class

        pred=model.predict(image)
        class_idx=np.argmax(pred,axis=1)[0]
        result=classes[class_idx]

        

        return render_template('index.html',prediction='Prediction: {}'.format(result))


       
if __name__=="__main__":
    app.run(debug=True)
