from PIL import Image, ImageFilter # Para el resizing
# Optimizador
from scipy.ndimage.measurements import mean
import numpy as np
import cv2
from firebase_admin import storage
import os
import math




# FUNCIONES
# en load_keras_model() cambiar el path al modelo .keras
def load_keras_model():
    # Lazy load TensorFlow imports
    from tensorflow.keras.models import load_model
    
    global model
    print("* Loading model...")
    model = load_model('./keras/model.keras', compile=False)
    print("* Model loaded")
    
def prepare_image(path):
    # Lazy load TensorFlow imports
    from tensorflow.keras.utils import load_img, img_to_array
    
    img = img_to_array(load_img(path, target_size=(80, 128), color_mode="grayscale"))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    keras_new_predictions = model.predict(img)
    return keras_new_predictions

def get_overlay(hash):
    original = cv2.imread("images/image.png")
    mask = cv2.imread("images/masked.png", cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
    
    # Fix: Resize mask to match original image dimensions
    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    roi = cv2.bitwise_or(original, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))  # Convert mask for bitwise OR
    img1_color = original

    # Create a mask where the grayscale mask is greater than 0
    mask_bool = mask > 0

    # Create a red-colored version of the ROI (3 channels)
    red_thresh = np.zeros_like(img1_color)
    red_thresh[mask_bool] = [0, 0, 255]  # Red color in BGR format

    # Define transparency
    alpha = 0.5

    # Overlay with transparency
    img1_masked = img1_color.copy()
    img1_masked[mask_bool] = cv2.addWeighted(img1_color[mask_bool], 1.0, red_thresh[mask_bool], alpha, 0)

    # Convert to RGB and save
    overlay_rgb = cv2.cvtColor(img1_masked, cv2.COLOR_BGR2RGB)
    cv2.imwrite("images/overlay.png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    fileName = "overlays/{}.png".format(hash)
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename("images/overlay.png")
    blob.make_public()
    public_url = blob.public_url
    return public_url

def resize_and_smooth(img_array):
    # Asegúrarse de que el array tiene valores entre 0 y 255
    if img_array.max() <= 1:
        img_array = (img_array * 255).astype(np.uint8)  # Si los valores están en el rango [0, 1]
    else:
        img_array = img_array.astype(np.uint8)
    # Resizing
    try:
      img_array = cv2.resize(img_array, dsize=(2448, 1530), interpolation=cv2.INTER_CUBIC)
    except: print("ERROR")
    # Smoothing
    image = Image.fromarray(img_array)
    image = image.filter(ImageFilter.ModeFilter(size=50))
    return image

def save_image_from_array(img_array, save_path, file_name, hash):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists("masks"):
        os.makedirs("masks")
    img_array.save(os.path.join(save_path, file_name))
    fileName = "masks/{}.png".format(hash)
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename("images/masked.png")
    blob.make_public()
    public_url = blob.public_url
    overlay_url = get_overlay(hash)
    return {"mask_url": public_url, "overlay_url": overlay_url}

# Measure thickness
def measure(img_array):

    conversion_factor = 71.0

    points2 = [(0, 0), (0, 0)]
    for i in range(0, img_array.shape[0]):
        for j in range(1, img_array.shape[1]):
            if img_array[i,j] == 255:
                points2[0] = (j, i)
                break
    for i in range(img_array.shape[0]-1, 0, -1):
        for j in range(img_array.shape[1]-1, 0, -1):
            if img_array[i,j] == 255:
                points2[1] = (j, i)
                break

    distance = np.sqrt((points2[0][0] - points2[1][0])**2 + (points2[0][1] - points2[1][1])**2)

    delta_x = points2[0][0] - points2[1][0]
    delta_y = points2[0][1] - points2[1][1]

    theta_radians = math.atan2(delta_y, delta_x)
    theta_degrees = math.degrees(theta_radians)

    def rotate_image(image, angle):
          image_center = tuple(np.array(image.shape[1::-1]) / 2)
          rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
          result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
          return result

    rotated = rotate_image(img_array, theta_degrees)

    points3 = [(0, 0), (0, 0)]
    for i in range(0, rotated.shape[1]):
        for j in range(0, rotated.shape[0]):
            if rotated[j, i] == 255:
                points3[0] = (i, j)
                break

    for i in range(rotated.shape[1]-1, 0, -1):
        for j in range(0, rotated.shape[0]):
            if rotated[j, i] == 255:
                points3[1] = (i, j)
                break

    half_point = (int((points3[0][0] + points3[1][0]) / 2), int((points3[0][1] + points3[1][1]) / 2))

    points2 = [(0, 0), (0, 0)]

    for i in range(half_point[1], rotated.shape[0]):
        if rotated[i, half_point[0]] == 0:
            points2[0] = (half_point[0], i)
            break

    for i in range(half_point[1], 0, -1):
        if rotated[i, half_point[0]] == 0:
            points2[1] = (half_point[0], i)
            break

    distance2 = np.sqrt((points2[0][0] - points2[1][0])**2 + (points2[0][1] - points2[1][1])**2)

    return round(distance2 / conversion_factor, 2)

# Get echogenicity
def predict_class(og, mask):

  # LESION INTENSITY
  og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
  LESION_INTENSITY = mean(og_gray, mask)

  # RETINA INTENSITY
  newMask = np.zeros((og.shape[0], og.shape[1]), dtype=np.uint8)
  for i in range(0, mask.shape[0]):
      for j in range(1, mask.shape[1]):
          if mask[i, j-1] == 255 and mask[i, j] == 0:
              for k in range(0, 180):
                  newMask[i, j + k] = 255
  og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
  RETINA_INTENSITY = mean(og_gray, newMask)

  tolerance = 0.27 * LESION_INTENSITY
  difference = abs(LESION_INTENSITY - RETINA_INTENSITY)

  if difference <= tolerance:
      class_result = 'Hiper-ecogénico'
  else:
      class_result = 'Hipo-ecogénico'

  print(f"PRED:{class_result}")

  return class_result