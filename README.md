# YOLOv5 Face Recognition
This application fine-tunes a YOLOv5 model to recognize faces and objects in images.
 Prerequisites

  Docker

## Usage

### Build the Docker image:

bash
Copy
```
docker build -t yolov5_face_recognition .
```
Run the Docker container:

bash
Copy
```docker run -v $(pwd)/output:/app/output yolov5_face_recognition <URL_TO_ZIPPED_FILE> <PERSON_NAME>
```
Replace <URL_TO_ZIPPED_FILE> with the URL of the zipped file containing face images, and <PERSON_NAME> with the name of the person in the images.
The fine-tuned model will be saved in the output directory with the name <PERSON_NAME>_model.pt.
### Output
The application will print the results of object detection and face recognition for each image in the input zip file.
Notes

**This implementation uses YOLOv5 from Ultralytics.**
The model is fine-tuned for 100 epochs by default. You can adjust this in the fine_tune_model function in utils.py.
For better results, you may need to annotate your dataset properly and adjust the training parameters.

Performance Metrics
The YOLOv5 training process outputs various performance metrics, including mAP (mean Average Precision), precision, and recall. These are logged during the training process.
Optimizations
To reduce fine-tuning time:

Use a smaller YOLOv5 model (e.g., YOLOv5n instead of YOLOv5s)
Reduce the number of epochs
Use transfer learning by freezing some layers of the model

Retaining Original Capabilities
The fine-tuned model retains its original object detection capabilities while adding the ability to recognize the new face class.
