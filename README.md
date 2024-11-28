# REST API Service for Mmy App

This is the backend service for the **Mmy App CopEyeLot** component, developed by:

- **Rogelio Eduardo Benavides De La Rosa**
- **Alan García Rodríguez**
- **Joel Ángel López Plata**

This REST API provides two endpoints: `receive_pdf` and `receive_image`.

---

## Endpoints Overview

### 1. **Endpoint: `receive_pdf`**

#### **Description**  
Extracts all images from a PDF file, stores them in Firebase Storage, and returns a list containing links to the stored images.

#### **Request**  
- **Input**: A URL pointing to a PDF file.

#### **Response**  
- **Output**: An array of links to the extracted images.

---

### 2. **Endpoint: `receive_image`**

#### **Description**  
Analyzes an image to locate a tumor, measures its dimensions, classifies its echogenicity, and stores the results in Firestore. The response includes the original image, a generated mask, an overlay of the mask on the original image, the measured width of the tumor, and its echogenicity classification.

#### **Request**  
- **Input**: A URL pointing to an image file.

#### **Response**  
- **Output**: A JSON object containing:  
  - Link to the original image.  
  - Link to the generated mask.  
  - Link to the overlay image (mask on the original).  
  - Measured width of the melanoma.  
  - Echogenicity classification.  
