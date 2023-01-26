import PyQt5

import funtestgui

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MTRC Ground Control")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.roll_label = QLabel("Roll: ", self)
        self.pitch_label = QLabel("Pitch: ", self)
        self.yaw_label = QLabel("Yaw: ", self)
        self.altitude_label = QLabel("Altitude: ", self)
        # create a text label

        # create a vertical box layout and add the two labels
        grid = QGridLayout()
        grid.addWidget(self.image_label, 0, 0)

        vbox = QVBoxLayout()
        vbox.addWidget(self.roll_label)
        vbox.addWidget(self.pitch_label)
        vbox.addWidget(self.yaw_label)
        vbox.addWidget(self.altitude_label)
        wrapper = QWidget()
        wrapper.setLayout(vbox)
        grid.addWidget(wrapper, 0, 1)

        # set the vbox layout as the widgets layout
        self.setLayout(grid)

        # create the video capture thread
        print("Create thread")
        self.thread = funtestgui.VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        print("Thread started")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
app = QApplication(sys.argv)
a = App()
a.show()
sys.exit(app.exec_())