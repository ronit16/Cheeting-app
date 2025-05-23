import os
import math
from collections import Counter
from google.cloud import vision
import re
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'key.json'

def extract_text_from_image(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []
    
    if texts:
        return texts[0].description.strip()
    else:
        return "No text detected"
    
image_path = "/home/neel/Desktop/HyperLink/OCR/shared image1.jpeg"
text = extract_text_from_image(image_path)
print(text)