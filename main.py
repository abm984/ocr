import io
import os
import json
from fastapi import FastAPI, UploadFile, HTTPException
from google.cloud import vision

app = FastAPI()

# Set up
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'noonan-389919-03321b6e11a4.json'
client = vision.ImageAnnotatorClient()

def perform_ocr(image_content):
    # image object
    vision_image = vision.Image(content=image_content)

    # text detection
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations

    result = {
        'text': texts[0].description,
        'vertices': []
    }

    # bounding box 
    for text in texts[1:]:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        result['vertices'].append(vertices)

    return result

def save_as_json(result, output_file):
    with open(output_file, 'w') as file:
        json.dump(result, file)

def save_as_text(result, output_file):
    with open(output_file, 'w') as file:
        file.write(result['text'])

@app.post("/ocr")
async def ocr(image: UploadFile = None):
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided.")

    # Read 
    image_content = await image.read()

    #  OCR 
    result = perform_ocr(image_content)

    # JSON file
    json_output_file = f"{image.filename.split('.')[0]}.json"
    save_as_json(result, json_output_file)

    # text document
    text_output_file = f"{image.filename.split('.')[0]}.txt"
    save_as_text(result, text_output_file)

    return result