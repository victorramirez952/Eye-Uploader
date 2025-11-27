from firebase_functions import https_fn, options
from firebase_admin import storage
import json
import requests
import hashlib
import os
import shutil
from spire.pdf import PdfDocument, PdfImageHelper
from concurrent.futures import ThreadPoolExecutor, as_completed


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

        print(f"Affected Eye images: {len(affected_eye_images)}")
        
        # Upload only Affected Eye images to Firebase Storage
        upload_urls = []
        
        # OPTIMIZATION: Use ThreadPoolExecutor for parallel uploads
        def upload_single_image(image_path):
            try:
                # Read file content once for both hashing and uploading
                with open(image_path, "rb") as f:
                    content = f.read()
                
                hash = hashlib.sha256(content).hexdigest()
                
                fileName = "tempImages/{}.png".format(hash)
                bucket = storage.bucket()
                blob = bucket.blob(fileName)
                blob.content_type = 'image/png'
                
                # Upload from memory to avoid re-reading from disk
                blob.upload_from_string(content, content_type='image/png')

                # Make public access from the URL
                blob.make_public()
                public_url = blob.public_url
                return public_url
            except Exception as e:
                print(f"Error uploading image {image_path}: {str(e)}")
                return None

        # Use 10 workers for parallel uploads (Network I/O bound)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_path = {executor.submit(upload_single_image, path): path for path in affected_eye_images}
            
            for future in as_completed(future_to_path):
                url = future.result()
                if url:
                    upload_urls.append(url)
        
        # Clean up tempImages directory
        shutil.rmtree("tempImages")
        
        return upload_urls
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Clean up in case of error
        if os.path.exists("tempImages"):
            shutil.rmtree("tempImages")
        return []


@https_fn.on_request(
    timeout_sec=240,
    memory=options.MemoryOption.GB_4,
    cpu=4,
    preserve_external_changes=True,
    cors=options.CorsOptions(
        cors_origins=["*"],
        cors_methods=["get", "post"],
    ))
def receive_pdf(req: https_fn.Request) -> https_fn.Response:
    try:
        body_data = req.get_data().decode('utf-8').strip() 
        # Get the body data as bytes and decode it to a string
        body_json = json.loads(body_data)
        print("Received request data for pdf endpoint:", body_json)
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
