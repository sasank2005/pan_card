import os
import cv2
import requests
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import imutils

def detect_tampering(original_url, tampered_url, output_dir="pan_card_tampering/image"):
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images from URLs
    original_resp = requests.get(original_url, stream=True).raw
    tampered_resp = requests.get(tampered_url, stream=True).raw
    
    original = Image.open(original_resp)
    tampered = Image.open(tampered_resp)
    
    # Resize images to a common size
    new_size = (250, 160)
    original = original.resize(new_size)
    tampered = tampered.resize(new_size)
    
    # Save resized images
    original_path = os.path.join(output_dir, "original.png")
    tampered_path = os.path.join(output_dir, "tampered.png")
    original.save(original_path)
    tampered.save(tampered_path)
    
    # Read images using OpenCV
    original = cv2.imread(original_path)
    tampered = cv2.imread(tampered_path)
    
    # Convert to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM Score:", score)
    
    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Draw bounding boxes around differences
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Save output images
    diff_image_path = os.path.join(output_dir, "diff.png")
    cv2.imwrite(diff_image_path, diff)
    
    # Return the SSIM score and the image paths
    return score, [original_path, tampered_path, diff_image_path]