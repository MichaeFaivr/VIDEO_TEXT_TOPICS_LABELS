import torch
from PIL import Image
import cv2
import numpy as np

# class  for one image framing and labelling
# voir comment utiliser *args et **kwargs

# voir avec ChatGPT comment détecter des objets d'une liste d'objets donnée

class FrameClass():
    def __init__(self,path,list_obj_types):
        self.path = path
        self.list_obj_types = list_obj_types
        self.yolo_model_path = 'ultralytics/yolov5'

    def YOLO_model(self):
        self.model = torch.hub.load(self.yolo_model_path, 'yolov5s')

    def build_results(self):
        # Perform inference
        results = self.model(self.path)
        # Print results
        results.print()  # Prints the detected objects with confidence scores
        # Display the results
        results.show()  # Displays the image with bounding boxes
        # Save the results
        results.save()  # Saves the image with bounding boxes to 'runs/detect/exp'
        # Save as attribute
        self.results = results

    def detect_object(self):
        # parcourir les types d'objets de la liste
        # frame objects
        # labelling
        # proba 3 main types of objects
        # return detections
        self.detections = self.results.pandas().xyxy[0]  # Coordinates in format (xmin, ymin, xmax, ymax)
        print(self.detections)

    # Example to draw bounding boxes manually using OpenCV
    def draw_boxes(self, detections):
        img = cv2.imread(self.path)
        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image with bounding boxes
        cv2.imshow('Detected Objects', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_json_frames(self):
        pass

