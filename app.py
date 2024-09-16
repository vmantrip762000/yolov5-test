import argparse
import os
from utils import download_and_extract, load_yolo_model, fine_tune_model, save_model, detect_objects_and_faces

def main(url, person_name):
    # Download and extract images
    image_dir = download_and_extract(url)

    # Load pre-trained YOLOv5 model
    model = load_yolo_model()

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(model, image_dir, person_name)

    # Save the fine-tuned model
    model_path = os.path.join('output', f'{person_name}_model.pt')
    save_model(fine_tuned_model, model_path)

    # Perform object detection and face recognition
    results = detect_objects_and_faces(fine_tuned_model, image_dir)

    # Print results
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 Face Recognition')
    parser.add_argument('url', type=str, help='URL to the zipped file containing face images')
    parser.add_argument('person_name', type=str, help='Name of the person in the images')
    args = parser.parse_args()

    main(args.url, args.person_name)