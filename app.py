from flask import Flask, request, render_template, jsonify
import os
import cv2
import requests
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import imutils
from src.code import detect_tampering

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    original_url = request.form.get('original_url')
    tampered_url = request.form.get('tampered_url')
    
    score, image_paths = detect_tampering(original_url, tampered_url)
    
    return jsonify({
        "ssim_score": score,
        "images": image_paths
    })

if __name__ == '__main__':
    app.run(debug=True)