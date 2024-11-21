# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn, options
from firebase_admin import credentials, initialize_app, storage, firestore
from spire.pdf import PdfDocument, PdfImageHelper
import json
import requests
import hashlib
import cv2
import numpy as np
import os
import random
from PIL import Image, ImageFilter # Para el resizing
import tensorflow as tf
from computerVision import load_keras_model, prepare_image, predict, resize_and_smooth, save_image_from_array, measure, predict_class

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

cred = credentials.Certificate("./firebaseConfig.json")
initialize_app(cred, {'storageBucket': 'mmy-app-e9474.firebasestorage.app'})
db = firestore.client()

os.environ["ULTRASOUND"] = str(39)
np.random.seed(39) # Set seed. Now any random operation using NumPy will produce the same result
tf.random.set_seed(39) # # Now any random operation using TensorFlow will produce the same result
random.seed(39) # Now any random operation using the random module will produce the same result

def computer_vision(hash: str):
    load_keras_model()
    path = './images/image.png'
    # Cargar img original
    og = cv2.imread(path)
    # Preprocesar img
    og_prepro = prepare_image(path)
    # Predecir
    pred = np.squeeze(predict(og_prepro))
    # Resize y smooth de pred
    image = resize_and_smooth(pred) # resize
    mask = np.asarray(image)
    # Guardar mask bonita
    maskInfo = save_image_from_array(image, "./images/", 'masked.png', hash)
    
    # Medir
    thickness = measure(mask)
    # Clasificar
    pred_class = predict_class(og, mask)
    
    vision = {"mask": maskInfo["mask_url"], "overlay": maskInfo["overlay_url"], "width": thickness, "echogenicity": pred_class}
    return vision

def getImage(fileroute: str):
    try:
        # Create a PdfDocument instance
        pdf = PdfDocument()
        
        # Load a PDF file
        pdf.LoadFromFile(fileroute)
        
        # Create a PdfImageHelper instance
        imageHelper = PdfImageHelper()
        
        # Get the current page
        page = pdf.Pages.get_Item(0)
        
        # Get the image information of the page
        imageInfo = imageHelper.GetImagesInfo(page)
        
        imageInfo[1].Image.Save(f"images/image.png")
        
        # Save image to bucket
        with open("images/image.png", "rb") as f:
            hash = hashlib.sha256(f.read()).hexdigest()
        fileName = "images/{}.png".format(hash)
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename("images/image.png")

        # Opt : if you want to make public access from the URL
        blob.make_public()
        public_url = blob.public_url

        return {"hash": hash, "url": public_url}
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return ""
    
def uploadResults(data: dict, hash: str):
    try:
        doc_ref = db.collection(u'results').document(hash)
        doc_ref.set(data)
    except Exception as e:
        print(f"Error uploading results: {str(e)}")

# HTTP function that receives a body
@https_fn.on_request(cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def receive_image(req: https_fn.Request) -> https_fn.Response:
    try:
        # Get the body data as bytes and decode it to a string
        body_data = req.get_data().decode('utf-8')
        body_json = json.loads(body_data)
        image_link = body_json.get("link", "No image link provided")
        image = requests.get(image_link)
        open("pdfs/image.pdf", "wb").write(image.content)
        imageInformation = getImage("pdfs/image.pdf")
        checkIfExists = db.collection(u'results').document(imageInformation["hash"]).get()
        print("Check if exists", checkIfExists)
        if checkIfExists.exists:
            return https_fn.Response(response=json.dumps(checkIfExists.to_dict()), status=200)
        vision = computer_vision(imageInformation["hash"])
        body = {"image": imageInformation["url"], "mask": vision["mask"], "overlay": vision["overlay"], "pdf": image_link, "width": vision["width"], "echogenicity": vision["echogenicity"]}
        print(body)
        uploadResults(body, imageInformation["hash"])
        json_body = json.dumps(body)
        
        return https_fn.Response(response=json_body, status=200)
    except Exception as e:
        # Handle any errors that occur
        print(f"Error processing request: {str(e)}")
        return https_fn.Response(f"Error processing request: {str(e)}", status=400)