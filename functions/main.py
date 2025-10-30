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
from computerVision import load_keras_model, prepare_image, predict, resize_and_smooth, save_image_from_array, measure, predict_class

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

cred = credentials.Certificate("./firebaseConfig.json")
initialize_app(cred, {'storageBucket': 'eci-ot25.firebasestorage.app'})
db = firestore.client()

def computer_vision(hash: str):
    # Lazy load TensorFlow and set seeds only when needed
    import tensorflow as tf
    
    os.environ["ULTRASOUND"] = str(39)
    np.random.seed(39) # Set seed. Now any random operation using NumPy will produce the same result
    tf.random.set_seed(39) # # Now any random operation using TensorFlow will produce the same result
    random.seed(39) # Now any random operation using the random module will produce the same result
    
    load_keras_model()
    # Create images directory if not exist
    os.makedirs("images", exist_ok=True)
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
    
    # Resize mask to match original image dimensions for analysis
    mask_resized = cv2.resize(mask, (og.shape[1], og.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Guardar mask bonita (keep the high-res version for storage)
    maskInfo = save_image_from_array(image, "./images/", 'masked.png', hash)
    
    # Medir (use high-res mask for accurate measurements)
    thickness = measure(mask)
    # Clasificar (use resized mask that matches original image dimensions)
    pred_class = predict_class(og, mask_resized)
    
    vision = {"mask": maskInfo["mask_url"], "overlay": maskInfo["overlay_url"], "width": thickness, "echogenicity": pred_class}
    return vision

def getImages(fileroute: str):
    try:
        # Create tempImages directory if it doesn't exist
        os.makedirs("tempImages", exist_ok=True)
        
        # Create a PdfDocument instance
        pdf = PdfDocument()
        
        # Load a PDF file
        pdf.LoadFromFile(fileroute)
        
        # Create a PdfImageHelper instance
        imageHelper = PdfImageHelper()
        
        # Get number of pages
        pageCount = pdf.Pages.Count
        
        images = []
        imageIndex = 0
        
        for i in range(pageCount):
            # Get the current page
            page = pdf.Pages.get_Item(i)
            
            # Get the image information of the page
            imageInfo = imageHelper.GetImagesInfo(page)
            
            # Get number of images
            imageCount = len(imageInfo)
            for j in range(imageCount):
                imageInfo[j].Image.Save(f"tempImages/image{imageIndex}.png")
                
                # Save image to bucket
                with open("tempImages/image{}.png".format(imageIndex), "rb") as f:
                    hash = hashlib.sha256(f.read()).hexdigest()
                fileName = "tempImages/{}.png".format(hash)
                bucket = storage.bucket()
                blob = bucket.blob(fileName)
                blob.upload_from_filename(f"tempImages/image{imageIndex}.png")

                # Opt : if you want to make public access from the URL
                blob.make_public()
                public_url = blob.public_url

                images.append(public_url)
                imageIndex += 1
        return images
        
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
def receive_pdf(req: https_fn.Request) -> https_fn.Response:
    try:
        body_data = req.get_data().decode('utf-8').strip() 
        # Get the body data as bytes and decode it to a string
        body_json = json.loads(body_data)
        image_link = body_json.get("link", "No image link provided")
        image = requests.get(image_link)
        # Get pdfs directory if not exist
        os.makedirs("pdfs", exist_ok=True)
        open("pdfs/image.pdf", "wb").write(image.content)
        return json.dumps(getImages("pdfs/image.pdf"))
    except Exception as e:
        # Handle any errors that occur
        print(f"Error processing request: {str(e)}")
        return https_fn.Response(f"Error processing request: {str(e)}", status=400)

# HTTP function that receives a body
@https_fn.on_request(cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def receive_image(req: https_fn.Request) -> https_fn.Response:
    try:
        body_data = req.get_data().decode('utf-8').strip() 
        # Get the body data as bytes and decode it to a string
        body_json = json.loads(body_data)
        image = requests.get(body_json["link"])
        os.makedirs("images", exist_ok=True)
        open("images/image.png", "wb").write(image.content)
        hash = hashlib.sha256(image.content).hexdigest()
        checkIfExists = db.collection(u'results').document(hash).get()
        if checkIfExists.exists:
            return https_fn.Response(response=json.dumps(checkIfExists.to_dict()), status=200)
        fileName = "images/{}.png".format(hash)
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename("images/image.png")
        blob.make_public()
        public_url = blob.public_url
        vision = computer_vision(hash)
        body = {"image": public_url, "mask": vision["mask"], "overlay": vision["overlay"], "width": vision["width"], "echogenicity": vision["echogenicity"]}
        uploadResults(body, hash)
        json_body = json.dumps(body)
        
        return https_fn.Response(response=json_body, status=200)
    except Exception as e:
        # Handle any errors that occur
        print(f"Error processing request: {str(e)}")
        return https_fn.Response(f"Error processing request: {str(e)}", status=400)

# Test endpoint to check Python version
@https_fn.on_request()
def python_version_eye_uploader(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response(f"Hello world\nPython version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test endpoint to verify TensorFlow import and version
@https_fn.on_request(cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def test_tensorflow(req: https_fn.Request) -> https_fn.Response:
    try:
        import tensorflow as tf
        version = tf.__version__
        print(f"TensorFlow version: {version}")
        response_data = {
            "status": "success",
            "tensorflow_version": version,
            "message": f"TensorFlow successfully imported. Version: {version}"
        }
        return https_fn.Response(response=json.dumps(response_data), status=200)
    except Exception as e:
        print(f"Error importing TensorFlow: {str(e)}")
        error_data = {
            "status": "error",
            "message": f"Failed to import TensorFlow: {str(e)}"
        }
        return https_fn.Response(response=json.dumps(error_data), status=500)