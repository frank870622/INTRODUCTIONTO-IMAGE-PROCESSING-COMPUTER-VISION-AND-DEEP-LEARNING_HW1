# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:01:39 2019

@author: 方嘉祥
"""

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import numpy as np
import cv2

class MainWindow (QMainWindow):
    def __init__ (self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('./F74056166.ui', self)
        self.Connect_btn()
        
        
    def Connect_btn(self):
        self.Btn_1_1.clicked.connect(self.Btn_1_1_function)
        self.Btn_1_2.clicked.connect(self.Btn_1_2_function)
        self.Btn_1_3.clicked.connect(self.Btn_1_3_function)
        self.Btn_1_4.clicked.connect(self.Btn_1_4_function)
        self.Btn_2_1.clicked.connect(self.Btn_2_1_function)
        self.Btn_2_2.clicked.connect(self.Btn_2_2_function)
        self.Btn_3_1.clicked.connect(self.Btn_3_1_function)
        self.Btn_3_2.clicked.connect(self.Btn_3_2_function)
        self.Btn_4_1.clicked.connect(self.Btn_4_1_function)
        self.Btn_4_2.clicked.connect(self.Btn_4_2_function)
        self.Btn_4_3.clicked.connect(self.Btn_4_3_function)
        self.Btn_4_4.clicked.connect(self.Btn_4_4_function)
        self.Btn_5_1.clicked.connect(self.Btn_5_1_function)
        self.Btn_5_2.clicked.connect(self.Btn_5_2_function)
        self.Btn_5_3.clicked.connect(self.Btn_5_3_function)
        self.Btn_5_4.clicked.connect(self.Btn_5_4_function)
        self.Btn_5_5.clicked.connect(self.Btn_5_5_function)
        
    def Btn_1_1_function(self):
        img = cv2.imread('./images/dog.bmp')
        cv2.imshow("Load Image", img)
        
        print('Height: ' + str(img.shape[0]))
        print('Weight: ' + str(img.shape[1]))
        
    def Btn_1_2_function(self):
        img = cv2.imread('./images/color.png')
        img_change = np.zeros_like(img)
        img_change[..., 0] = img[..., 1]
        img_change[..., 1] = img[..., 2]
        img_change[..., 2] = img[..., 0]
        cv2.imshow("Color Image", img)
        cv2.imshow("Color Conversion", img_change)
        
    
    def Btn_1_3_function(self):
        img = cv2.imread('./images/dog.bmp')
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Image Flipping", img_flip)
        
    def Btn_1_4_function(self):
        img = cv2.imread('./images/dog.bmp')
        img_flip = cv2.flip(img, 1)
        
        def change_trackbar(value):
            img_blend = cv2.addWeighted(img_flip, value/100, img, 1-value/100, 0.0)
            cv2.imshow("Blending", img_blend)
        
        cv2.imshow("Blending", img)
        cv2.createTrackbar('Blend','Blending',0,100,change_trackbar)

        
    def Btn_2_1_function(self):
        print("Btn")
        
    def Btn_2_2_function(self):
        print("Btn")
        
    def Btn_3_1_function(self):
        print("Btn")
        
    def Btn_3_2_function(self):
        print("Btn")
        
    def Btn_4_1_function(self):
        print("Btn")
        
    def Btn_4_2_function(self):
        print("Btn")
        
    def Btn_4_3_function(self):
        print("Btn")
        
    def Btn_4_4_function(self):
        print("Btn")
        
    def Btn_5_1_function(self):
        print("Btn")
        
    def Btn_5_2_function(self):
        print("Btn")
        
    def Btn_5_3_function(self):
        print("Btn")
        
    def Btn_5_4_function(self):
        print("Btn")
        
    def Btn_5_5_function(self):
        print("Btn")
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

print('Hello')
