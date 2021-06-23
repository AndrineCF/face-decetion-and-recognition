import os
import json
import cv2
import numpy as np
from PIL import Image


class Trainer:
    def __init__(self):
        self.label_ids = {}
        self.x_train = []
        self.y_labels = []

        self.cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        self.haar_model_face = os.path.join(self.cv2_base_dir, 'data\haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(self.haar_model_face)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    def create_trainer(self):
        """
        It create the LBPH model and train on the dataset. The train file is stored and use when
        the program use face recognition.
        The code uses elements from
        https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/faces-train.py
        """
        current_id = 0

        path_root = os.getcwd().replace("\src", "")
        path_dataset = os.path.join(path_root, 'dataset')

        path_recognizers = os.path.join(path_root, 'recognizers')

        if not os.path.exists(path_recognizers):
            os.mkdir(path_recognizers)


        for root, dirs, files in os.walk(path_dataset):

            for file in files:

                path = os.path.join(root, file)
                label = os.path.basename(root)

                # check if the label are already exists
                if label not in self.label_ids:
                    self.label_ids[label] = current_id
                    current_id += 1

                id_ = self.label_ids[label]

                pil_image = Image.open(path).convert("L")  # grayscale
                image_array = np.array(pil_image, "uint8")

                faces = self.face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    self.x_train.append(roi)
                    self.y_labels.append(id_)

        with open(f"{path_recognizers}/face_labels.json", 'w') as f:
            json.dump(self.label_ids, f, indent=4, sort_keys=True)

        try:
            self.recognizer.train(self.x_train, np.array(self.y_labels))
            self.recognizer.write(f"{path_recognizers}/face_trainer.yml")
        except Exception as e:
            print(e)
