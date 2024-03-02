# Import necessary libraries
from fastapi import FastAPI, File, UploadFile,Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from keras.preprocessing import image
import pickle
from flask import Flask, request
import io
import numpy as np

# Load the trained model and other necessary objects
with open('Brain_pickle (1).pkl', 'rb') as file:
    model = pickle.load(file)
# Create a FastAPI app
app = FastAPI()

# Create an instance of the Jinja2Templates class for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Define the root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the endpoint to handle image uploads and make predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Load and preprocess the image
    img = image.load_img(io.BytesIO(contents), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to the range [0, 1]

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Map class index to class name
    class_folders = ['glioma', 'meningioma', 'pituitary', 'notumor']
    predicted_class_name = class_folders[predicted_class]

    # Render the result template
    return templates.TemplateResponse("result.html", {"request": request, "prediction": predicted_class_name})
