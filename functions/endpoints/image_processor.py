from firebase_functions import https_fn, options
from firebase_admin import storage, firestore
import json
import requests
import hashlib
import cv2
import numpy as np
import os
import shutil
import random
from PIL import Image


db = firestore.client()


def computer_vision(hash: str, verbose = True):
    from enhanced_measurements import Measurer
    from enhanced_measurements.config import MEDIAN_RULE_MM, MEDIAN_AP_AXIS
    from measure_enhanced import measure, visualize_measure
    from computerVision import load_keras_model, prepare_image, predict, resize_and_smooth, save_image_from_array, predict_class
    
    os.environ["ULTRASOUND"] = str(39)
    np.random.seed(39)
    random.seed(39)
    
    load_keras_model()
    # Create images directory if not exist
    os.makedirs("images", exist_ok=True)
    path = './images/image.png'
    # Hash of image
    image_hash = hashlib.sha256(open(path, 'rb').read()).hexdigest()

    # Cargar img original
    og = cv2.imread(path)
    # Preprocesar img
    og_prepro = prepare_image(path)
    # Predecir
    pred = np.squeeze(predict(og_prepro))
    # Resize y smooth de pred
    image = resize_and_smooth(pred)
    mask = np.asarray(image)
    
    # Resize mask to match original image dimensions for analysis
    mask_resized = cv2.resize(mask, (og.shape[1], og.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Guardar mask bonita (keep the high-res version for storage)
    maskInfo = save_image_from_array(image, "./images/", 'masked.png', hash)
    
    # Use Measurer to get mm_per_pixel calibration
    equivalencies = {
        'median_rule_mm': MEDIAN_RULE_MM,
        'median_ap_axis': MEDIAN_AP_AXIS
    }
    
    # Save the resized mask temporarily for Measurer
    mask_resized_path = './images/mask_resized.png'
    cv2.imwrite(mask_resized_path, mask_resized)
    
    measurer = Measurer(temp_folder="tempImages", verbose=verbose)
    result = measurer.process_image(
        path,
        mask_path=mask_resized_path,
        equivalencies=equivalencies,
        test_mode=False
    )
    
    mm_per_pixel = result['mm_per_pixel'] if result else 0
    mm_per_pixel -= 0.00128  # Calibration offset
    method = result['method'] if result else 'none'
    
    # Use measure to get the thickness and basal diameter measurements in mm
    measurements = measure(mask_resized, mm_per_pixel)
    thickness = measurements['thickness']
    basal_diameter = measurements['basal_diameter']
    
    # Generate visualization if verbose
    if verbose:
        visualize_measure(mask_resized, mm_per_pixel, measurements)
    
    # Classify echogenicity using resized mask that matches original image dimensions
    pred_class = predict_class(og, mask_resized)
    
    vision = {"mask": maskInfo["mask_url"], "overlay": maskInfo["overlay_url"], "width": thickness, "basal_diameter": basal_diameter, "echogenicity": pred_class}
    return vision


def uploadResults(data: dict, hash: str):
    try:
        doc_ref = db.collection(u'results').document(hash)
        doc_ref.set(data)
    except Exception as e:
        print(f"Error uploading results: {str(e)}")


@https_fn.on_request(
    timeout_sec=240,
    memory=options.MemoryOption.GB_2,
    cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def receive_image(req: https_fn.Request) -> https_fn.Response:
    try:
        body_data = req.get_data().decode('utf-8').strip() 
        # Get the body data as bytes and decode it to a string
        body_json = json.loads(body_data)
        print("Received request data for image endpoint:", body_json)
        image = requests.get(body_json["link"])
        # Clean up images directory if it exists
        if os.path.exists("images"):
            shutil.rmtree("images")
        os.makedirs("images", exist_ok=True)
        open("images/image.png", "wb").write(image.content)
        hash = hashlib.sha256(image.content).hexdigest()
        checkIfExists = db.collection(u'results').document(hash).get()
        
        # Compatibility fix: remove entries without 'basal_diameter'
        if checkIfExists.exists and 'basal_diameter' not in checkIfExists.to_dict():
            db.collection(u'results').document(hash).delete()
            print("Deleted incomplete entry for hash:", hash)
        elif checkIfExists.exists and 'basal_diameter' in checkIfExists.to_dict():
            return https_fn.Response(response=json.dumps(checkIfExists.to_dict()), status=200)

        fileName = "images/{}.png".format(hash)
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.content_type = 'image/png'
        blob.upload_from_filename("images/image.png")
        blob.make_public()
        public_url = blob.public_url
        vision = computer_vision(hash)
        body = {"image": public_url, "mask": vision["mask"], "overlay": vision["overlay"], "width": vision["width"], "basal_diameter": vision["basal_diameter"], "echogenicity": vision["echogenicity"]}
        uploadResults(body, hash)
        json_body = json.dumps(body)
        
        return https_fn.Response(response=json_body, status=200)
    except Exception as e:
        # Handle any errors that occur
        print(f"Error processing request: {str(e)}")
        return https_fn.Response(f"Error processing request: {str(e)}", status=400)


@https_fn.on_request(
    timeout_sec=240,
    memory=options.MemoryOption.GB_16,
    cpu=4,
    preserve_external_changes=True,
    cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def receive_image_gpu(req: https_fn.Request) -> https_fn.Response:
    try:
        body_data = req.get_data().decode('utf-8').strip() 
        # Get the body data as bytes and decode it to a string
        body_json = json.loads(body_data)
        print("Received request data for image GPU endpoint:", body_json)
        image = requests.get(body_json["link"])
        # Clean up images directory if it exists
        if os.path.exists("images"):
            shutil.rmtree("images")
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
        body = {"image": public_url, "mask": vision["mask"], "overlay": vision["overlay"], "width": vision["width"], "basal_diameter": vision["basal_diameter"], "echogenicity": vision["echogenicity"]}
        uploadResults(body, hash)
        json_body = json.dumps(body)
        
        return https_fn.Response(response=json_body, status=200)
    except Exception as e:
        # Handle any errors that occur
        print(f"Error processing request: {str(e)}")
        return https_fn.Response(f"Error processing request: {str(e)}", status=400)
