# REST API Service for Eye Uploader

Backend service for medical ultrasound image analysis, originally developed by:

- **Rogelio Eduardo Benavides De La Rosa**
- **Alan García Rodríguez**
- **Joel Ángel López Plata**

---

## API Endpoints

### `receive_pdf`

Extracts ultrasound images from PDF documents and filters them to identify affected eye images using automated classification.

**Request**
```json
{
  "link": "https://example.com/document.pdf"
}
```

**Response**
```json
[
  "https://storage.googleapis.com/...",
  "https://storage.googleapis.com/..."
]
```

**Process**
1. Downloads PDF from provided URL
2. Extracts all images from the document
3. Classifies images to identify affected eye ultrasounds
4. Uploads filtered images to Firebase Storage
5. Returns public URLs of classified images

---

### `receive_image`

Analyzes ultrasound images to detect and measure ocular lesions, classify echogenicity, and generate visual overlays.

**Request**
```json
{
  "link": "https://example.com/ultrasound.png"
}
```

**Response**
```json
{
  "image": "https://storage.googleapis.com/original.png",
  "mask": "https://storage.googleapis.com/mask.png",
  "overlay": "https://storage.googleapis.com/overlay.png",
  "thickness": 2.45,
  "echogenicity": "Hipo-ecogénico"
}
```

**Process**
1. Downloads ultrasound image from provided URL
2. Uses deep learning model to generate lesion segmentation mask
3. Calculates mm-per-pixel calibration using ruler or anatomical landmarks
4. Measures lesion thickness in millimeters using perpendicular distance method
5. Classifies echogenicity by comparing lesion and retina intensities
6. Generates overlay visualization with measurement annotations
7. Stores results in Firestore and returns URLs with measurements

**Measurement Method**

The thickness measurement uses an adaptive perpendicular distance algorithm that traces along the lesion edge and calculates the maximum perpendicular distance within the mask boundaries.

**Echogenicity Classification**

- **Hiper-ecogénico**: Lesion intensity similar to retina (within 27% tolerance)
- **Hipo-ecogénico**: Lesion intensity significantly different from retina

---

### `tridimensional_reconstruction`

Generates 3D mesh models from transversal and longitudinal ultrasound mask images using extrusion-based reconstruction.

**Request**
```json
{
  "transversal_image_url": "https://example.com/transversal_mask.png",
  "longitudinal_image_url": "https://example.com/longitudinal_mask.png",
  "base_T": 5.2,
  "base_L": 6.8,
  "height": 4.5
}
```

**Response**
```json
{
  "status": "success",
  "glb_url": "https://storage.googleapis.com/model.glb",
  "area_mm2": 35.36,
  "volume_mm3": 158.62,
  "processing_time": 2.34
}
```

**Process**
1. Downloads transversal and longitudinal mask images
2. Extracts 2D contours from both views
3. Calibrates measurements using provided dimensions (mm)
4. Performs extrusion-based 3D reconstruction
5. Calculates surface area and volume
6. Exports mesh as GLB format
7. Uploads to Firebase Storage and returns metrics

**Parameters**
- `base_T`: Basal thickness of transversal image in millimeters
- `base_L`: Basal length of longitudinal image in millimeters
- `height`: Height measurement in millimeters

---

## Technical Details

**Technologies**
- Firebase Cloud Functions (Python 3.11)
- TensorFlow/Keras for deep learning inference
- OpenCV for image processing
- NumPy for numerical operations

**Resource Configuration**
- `receive_pdf`: 4GB memory, 2 CPU cores, 240s timeout
- `receive_image`: 2GB memory, 240s timeout
- `tridimensional_reconstruction`: 1GB memory, 300s timeout

**CORS Policy**

All endpoints accept requests from any origin with GET and POST methods enabled.  
