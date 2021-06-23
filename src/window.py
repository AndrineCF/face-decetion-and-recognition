import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFormLayout
from src.video import VideoThread
from trainer import Trainer


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.path_root = os.getcwd().replace("\src", "")

        self.dark_theme()
        self.setup_gui()

    def setup_gui(self):
        """
        Setup for the gui layout
        """
        self.layout_cam = QVBoxLayout()

        self.layout_form_butttons = QFormLayout()

        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)

        # Start button
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)

        # Recognition button
        self.detection_btn = QPushButton("Detection")
        self.detection_btn.clicked.connect(self.detection)

        # Recognition button
        self.recognition_btn = QPushButton("Recognition")
        self.recognition_btn.clicked.connect(self.recognition)

        self.layout_form_butttons.addRow(self.start_btn, self.stop_btn)
        self.layout_form_butttons.addRow(self.recognition_btn, self.detection_btn)

        self.layout_cam.addLayout(self.layout_form_butttons)

        self.webcam_label = QLabel()
        self.layout_cam.addWidget(self.webcam_label)

        self.cam = VideoThread()

        self.layout_form_save_image = QFormLayout()

        # Saved button and text input
        self.saved_image_btn = QPushButton("Saved image")
        self.saved_image_btn.clicked.connect(lambda: self.save())
        self.name_image_name = QLabel()
        self.name_image_name.setText("Name:")
        self.input_image_name = QLineEdit()
        self.layout_form_save_image.addRow(self.name_image_name, self.input_image_name)
        self.layout_form_save_image.addWidget(self.saved_image_btn)

        self.layout_cam.addLayout(self.layout_form_save_image)

        self.start()

        self.setLayout(self.layout_cam)

    def dark_theme(self):
        """
        Set a dark theme to the gui
        """
        path = os.path.join(self.path_root, 'stylesheet\stylesheet.css')
        stylesheet = open(path, "r")
        stylesheet = stylesheet.read()
        self.setStyleSheet(stylesheet)

    def update_webcam(self, frame):
        """
        Set thye webcam fram to the gui
        """
        self.webcam_label.setPixmap(QPixmap.fromImage(frame))

    def stop(self):
        """
        It stop the webcam frame
        """
        self.cam.face_detection_is_on = False
        self.cam.face_recognition_is_on = False
        self.cam.stop()

    def start(self):
        """
        It start the webcam frame
        """
        self.cam.start()
        self.cam.image_update.connect(self.update_webcam)

    def detection(self):
        if not self.cam.face_detection_is_on:
            self.cam.face_detection_is_on = True
        else:
            self.cam.face_detection_is_on = False

    def recognition(self):
        if not self.cam.face_recognition_is_on:
            try:
                self.trainer = Trainer()
                self.trainer.create_trainer()
                self.cam.face_recognition_is_on = True
            except:
                print("shit")
        else:
            self.cam.face_recognition_is_on = False

    def save(self):
        file_name = self.input_image_name.text().replace(" ", "_").lower()
        self.cam.saved_image(file_name)
