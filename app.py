import gradio as gr
import tensorflow as tf
import numpy as np
import os
import pandas as pd

model_gender = tf.keras.models.load_model('model_gender.h5')
model_age = tf.keras.models.load_model('model_age.h5')

actual_data = {
    "000000.png": {"img": 1,"age": 85.0, "gender": "female"},
    "000001.png": {"img": 2,"age": 72.0, "gender": "female"},
    "000002.png": {"img": 3,"age": 45.0, "gender": "male"},
    "000003.png": {"img": 4,"age": 59.0, "gender": "male"},
    "000004.png": {"img": 5,"age": 37.0, "gender": "male"}
    }

df = pd.DataFrame(actual_data).T

def preprocess_image(image):
  # Assuming image is a PIL Image object from Gradio
  img = image.convert('L')  # Convert to grayscale
  img = img.resize((128, 128))
  img = np.array(img) / 255.0  # Normalize pixel values
  img = img.reshape((1, 128, 128, 1))  # Add channel dimension
  return img

def predict(image):
  preprocessed_image = preprocess_image(image)
  gender_pred = model_gender.predict(preprocessed_image)[0][0]
  age_pred = model_age.predict(preprocessed_image)[0][0]
  gender = "Male" if gender_pred > 0.5 else "Female"
  list = "{:.2f}".format(age_pred),gender,df
  return list



# Gradio Interface with separate outputs
text_age = gr.components.Textbox(label="Predicted Age")
text_gender = gr.components.Textbox(label="Predicted Gender")

interface = gr.Interface(predict, gr.components.Image(height=440,width=1000,label="Upload Image", type="pil"),
                          outputs=[text_age, text_gender, gr.DataFrame(value=df)],
                          examples=[
                            os.path.join(os.path.dirname(__file__),"00000.png"),
                            os.path.join(os.path.dirname(__file__),"00001.png"),
                            os.path.join(os.path.dirname(__file__),"00002.png"),
                            os.path.join(os.path.dirname(__file__),"00003.png"),
                            os.path.join(os.path.dirname(__file__),"00004.png")],

                          allow_flagging='never', 
                          theme=gr.themes.Soft(),
                          title="Age and Gender Prediction").launch()


########### OUTDATED CODE #####################################
########### OUTDATED CODE #####################################
########### OUTDATED CODE #####################################
########### OUTDATED CODE #####################################
########### OUTDATED CODE #####################################
########### OUTDATED CODE #####################################
