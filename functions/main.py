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
import shutil
import random
from PIL import Image, ImageFilter # Para el resizing
from computerVision import load_keras_model, prepare_image, predict, resize_and_smooth, save_image_from_array, measure, predict_class

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Import 3D reconstruction module
from image_classifier.image_classifier import classify_directory

# cred = credentials.Certificate("./firebaseConfig.json")
cred = credentials.Certificate("C:\\Users\\jairr\\Documents\\UDEM\\9noSemestre\\PEF\\EYE_UPLOADER\\functions\\firebaseConfig.json")
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
        # Clean up tempImages directory if it exists
        if os.path.exists("tempImages"):
            shutil.rmtree("tempImages")
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
        
        imageIndex = 0
        
        # Extract all images from PDF to disk
        for i in range(pageCount):
            # Get the current page
            page = pdf.Pages.get_Item(i)
            
            # Get the image information of the page
            imageInfo = imageHelper.GetImagesInfo(page)
            
            # Get number of images
            imageCount = len(imageInfo)
            for j in range(imageCount):
                imageInfo[j].Image.Save(f"tempImages/image{imageIndex}.png")
                imageIndex += 1
        
        print(f"Extracted {imageIndex} images from PDF")
        
        affected_eye_images = []
        # Call image_classifier to get Affected Eye images
        from image_classifier.image_classifier import classify_directory
        affected_eye_images = classify_directory("tempImages")
        # image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        # for root, dirs, files in os.walk("tempImages"):
        #     for filename in files:
        #         if any(filename.endswith(ext) for ext in image_extensions):
        #             affected_eye_images.append(os.path.join(root, filename))
        
        print(f"Affected Eye images: {len(affected_eye_images)}")
        
        # Upload only Affected Eye images to Firebase Storage
        upload_urls = []
        for image_path in affected_eye_images:
            try:
                with open(image_path, "rb") as f:
                    hash = hashlib.sha256(f.read()).hexdigest()
                
                fileName = "tempImages/{}.png".format(hash)
                bucket = storage.bucket()
                blob = bucket.blob(fileName)
                blob.content_type = 'image/png'
                blob.upload_from_filename(image_path)

                # Make public access from the URL
                blob.make_public()
                public_url = blob.public_url

                upload_urls.append(public_url)
            except Exception as e:
                print(f"Error uploading image {image_path}: {str(e)}")
        
        # Clean up tempImages directory
        shutil.rmtree("tempImages")
        
        return upload_urls
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Clean up in case of error
        if os.path.exists("tempImages"):
            shutil.rmtree("tempImages")
        return []
    
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
        blob.content_type = 'image/png'
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

@https_fn.on_request()
def python_version_eye_uploader(req: https_fn.Request) -> https_fn.Response:
    import json
    print("Python version function called")
    
    # Handle CORS preflight request
    if req.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return https_fn.Response('', status=204, headers=headers)
    
    # Set CORS headers for actual request
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    response_data = {
        "message": "Hello world",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    return https_fn.Response(
        json.dumps(response_data),
        headers=headers
    )

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

# 3D Reconstruction endpoint
@https_fn.on_request(
    timeout_sec=300,
    memory=options.MemoryOption.GB_1,
    cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def tridimensional_reconstruction(req: https_fn.Request) -> https_fn.Response:
    """
    Endpoint for 3D reconstruction from ultrasound mask images.
    
    Parameters (from request body):
    - transversal_image_url: URL of the transversal mask image
    - longitudinal_image_url: URL of the longitudinal mask image
    - base_T: Measure of basal thickness of transversal image (mm)
    - base_L: Measure of basal length of longitudinal image (mm)
    - height: Height (mm)
    
    Returns:
    - JSON response with GLB file URL and measurements
    """
    try:
        # Parse request body
        body_data = req.get_data().decode('utf-8').strip()
        body_json = json.loads(body_data)
        
        # Extract parameters
        transversal_image_url = body_json.get("transversal_image_url")
        longitudinal_image_url = body_json.get("longitudinal_image_url")
        base_T = float(body_json.get("base_T"))
        base_L = float(body_json.get("base_L"))
        height = float(body_json.get("height"))
        
        # Validate parameters
        if not transversal_image_url or not longitudinal_image_url:
            return https_fn.Response(
                response=json.dumps({"error": "Missing image URLs"}),
                status=400
            )
        
        if base_T <= 0 or base_L <= 0 or height <= 0:
            return https_fn.Response(
                response=json.dumps({"error": "All measurements must be positive"}),
                status=400
            )
        
        # Perform 3D reconstruction
        from reconstruction_3d.extrusion_reconstruction import ExtrusionReconstruction
        reconstructor = ExtrusionReconstruction()
        
        result = reconstructor.reconstruct(
            transversal_image_url=transversal_image_url,
            longitudinal_image_url=longitudinal_image_url,
            base_t_mm=base_T,
            base_l_mm=base_L,
            h_mm=height
        )
        
        if result.get("success"):
            response_data = {
                "status": "success",
                "glb_url": result["glb_url"],
                "area_mm2": result["area_mm2"],
                "volume_mm3": result["volume_mm3"],
                "processing_time": result["processing_time"]
            }
            print(response_data)
            return https_fn.Response(
                response=json.dumps(response_data),
                status=200
            )
        else:
            return https_fn.Response(
                response=json.dumps({
                    "status": "error",
                    "message": result.get("error", "Unknown error during reconstruction")
                }),
                status=500
            )
        
    except Exception as e:
        print(f"Error in tridimensional_reconstruction: {str(e)}")
        return https_fn.Response(
            response=json.dumps({
                "status": "error",
                "message": str(e)
            }),
            status=500
        )
