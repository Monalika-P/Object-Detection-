# Importing the libraries
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

# Create an instance of the class ObjectDetection
detector = ObjectDetection()

# Calling the pre-trained models
detector.setModelTypeAsRetinaNet() # RetinaNet is the base model being used here
detector.setModelPath(os.path.join(execution_path, "D:\Object_detection\models\coco_best_resnet50_v2.0.1.h5"))

# Loading the models
detector.loadModel()

# Object Detection
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "D:\Object_detection\input\cd.jpg"),
                                             output_image_path=os.path.join(execution_path, "D:\Object_detection\output\cd.jpg"))

# Printing the dictionary containing name of each detected object along with its percentage probability
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])
