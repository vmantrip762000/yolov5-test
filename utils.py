import os
import requests
import zipfile
import torch
from ultralytics import YOLO
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

def download_and_extract(url):
    logging.info(f"Downloading and extracting images from {url}")
    response = requests.get(url)
    zip_path = 'images.zip'
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    image_dir = 'images'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(image_dir)
    
    os.remove(zip_path)
    logging.info(f"Extracted {len(os.listdir(image_dir))} images to {image_dir}")
    return image_dir

def load_yolo_model():
    logging.info("Loading pre-trained YOLOv5 model")
    model = YOLO('yolov5s.pt')
    return model

def fine_tune_model(model, image_dir, person_name):
    logging.info(f"Fine-tuning model for {person_name}")
    
    # Prepare dataset.yaml file
    dataset_yaml = f"""
    path: {image_dir}
    train: images
    val: images
    
    nc: 1
    names: ['{person_name}']
    """
    
    with open('dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    # Fine-tune the model
    results = model.train(data='dataset.yaml', epochs=100, imgsz=640)
    
    return model

def save_model(model, path):
    logging.info(f"Saving model to {path}")
    model.save(path)

def detect_objects_and_faces(model, image_dir):
    logging.info("Performing object detection and face recognition")
    results = []
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            prediction = model(image)
            results.append(f"Detected objects in {image_file}: {prediction}")
    
    return results