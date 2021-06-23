import json
import os
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class VideoThread(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self):
        super(VideoThread, self).__init__()

        # Set path to face detection, recognition and root
        self.cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        self.haar_model_face = os.path.join(self.cv2_base_dir, 'data\haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(self.haar_model_face)
        self.path_root = os.getcwd().replace("\src", "")

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.cam_is_on = True
        self.face_detection_is_on = False
        self.face_recognition_is_on = False

        # create directory if not already exists
        self.dataset_directory = self.create_directory("dataset")

    def run(self):
        """
        It runs the webcam
        """
        self.cam_is_on = True
        self.cam = cv2.VideoCapture(0)

        # check if cam is available
        if not self.cam.isOpened():
            print("No available webcam")
            self.cam_is_on = False

        while self.cam_is_on:
            self.ret, self.frame = self.cam.read()

            # Only run these function if boolean variable to these are true
            self.face_detection()
            self.face_recognition()

            if self.ret:
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                convert_to_qt_format = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.image_update.emit(convert_to_qt_format)

        self.cam.release()

    def stop(self):
        """
        It stop the thread and webcam
        """
        self.cam_is_on = False
        self.quit()

    def face_detection(self):
        """
        The function only detects faces
        """
        if self.face_detection_is_on:

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if self.ret:
                for (x, y, w, h) in faces:
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def face_recognition(self):
        """
        Face recognition function uses the train data from dataset and training file
        to recognition a person
        """

        if self.face_recognition_is_on:

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            path = os.path.join(self.path_root, "recognizers")

            try:
                recognizer.read(f"{path}\\face_trainer.yml")
            except Exception as e:
                print(e)

            with open(f"{self.path_root}\\recognizers\\face_labels.json", 'r') as jsonFile:
                json_object = json.load(jsonFile)

            labels = {}
            for key, value in json_object.items():
                labels[value] = key

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                try:
                    id_, conf = recognizer.predict(roi_gray)
                    if 60 <= conf <= 99:
                        name = labels[id_]
                        cv2.putText(self.frame, name, (x, y), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(self.frame, "Unknown", (x, y), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                except Exception as e:
                    print(e)

    def saved_image(self, name):
        """
        The function create a dataset directory for user if not already exists. Then stores images which
        are used to train the model to recognition faces
        :param name: name of a dataset directory to create (Assuming user write their name in)
        """
        dataset_directory_name = self.create_directory(f"{self.path_root}\dataset\{name}")

        path_dataset = os.path.join(self.path_root, "dataset")

        dir_list = os.listdir(path_dataset)

        try:
            if dataset_directory_name.split('\\')[-1] in dir_list:
                dir_list_dataset = os.listdir(dataset_directory_name)

                # it start on 1
                length_dataset = len(dir_list_dataset) + 1

                dataset_directory_file = f'{dataset_directory_name}\{length_dataset}.jpg'

                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                # Saving the image
                cv2.imwrite(dataset_directory_file, gray)

                print(f"Successfully creating {dataset_directory_file}")

        except Exception as e:
            print(e)

    def create_directory(self, name_directory):
        """
        It create a directory if not already exists
        :param name_directory: The name of directory
        :return: It return the path to the directory
        """
        path = os.path.join(self.path_root, name_directory)

        if not os.path.exists(path):
            os.mkdir(path)

        return path
