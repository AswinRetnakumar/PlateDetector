import numpy as np 
import pandas as pd 
import os 
import shutil
import torch
import cv2


class PlateDetector():

    def __init__(self, yolo_path = './yolov5/runs/train/'):
        
        latest_run = os.listdir(yolo_path)[-1]

        # Fetching the best weights 
        best_weights = os.path.join(yolo_path, latest_run, 'weights', 'best.pt')

        # Loading the model with best weights trained on custom data 
        self.model = torch.hub.load('./yolov5', 'custom', best_weights, source='local')

    def detect_plate(self, image):
        image = cv2.imread(image)
        # Convert image to RGB colorspace
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Predicting from model
        results = self.model(image)
        print(results.pandas().xyxy[0])
        results_df = results.pandas().xyxy[0]
        for i, result in results_df.iterrows():
            x_min = int(result['xmin'])
            x_max = int(result['xmax'])
            y_min = int(result['ymin'])
            y_max = int(result['ymax'])
            # Cropping license plate from image ""
            number_plate = image[y_min:y_max,x_min:x_max]
            name = 'plate'+str(i)+'.jpg'
            cv2.imwrite(name, number_plate)