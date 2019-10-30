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
import math

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
        img = cv2.imread('./images/QR.png')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_global = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY);
        
        cv2.imshow("QR img", img)
        cv2.imshow("Global Threshold", img_global)
        
        
    def Btn_2_2_function(self):
        img = cv2.imread('./images/QR.png')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_local = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1);
        
        cv2.imshow("QR img", img)
        cv2.imshow("Local Threshold", img_local)
        
    def Btn_3_1_function(self):
        
        angle = float(self.Angle_box.text())
        scale = float(self.Scale_box.text())
        tx = float(self.Tx_box.text())
        ty = float(self.Ty_box.text())
        
        img = cv2.imread('./images/OriginalTransform.png')
        transform_array = cv2.getRotationMatrix2D((tx, ty), angle, scale)
        img_change = cv2.warpAffine(img,transform_array,img.shape[:2])
        
        cv2.imshow("OriginalTransform", img)
        cv2.imshow("Transforms", img_change)
        
    def Btn_3_2_function(self):
        img = cv2.imread('./images/OriginalPerspective.png')
        img = cv2.resize(img, (800, 600), interpolation = cv2.INTER_AREA)
        
        mouse_click_point = []
        location = [(20, 20), (20, 450), (450, 450), (450, 20)]
        
        def mouse_click_left(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(mouse_click_point) < 4:
                mouse_click_point.append((x, y))
            
            if len(mouse_click_point) == 4:
                transform_array = cv2.getPerspectiveTransform(np.float32(mouse_click_point),np.float32(location))
                img_trans = cv2.warpPerspective(img,transform_array,(430,430))
                cv2.imshow("Perspective Transformation", img_trans)
                mouse_click_point.clear()
                
                
        cv2.imshow("OriginalPerspective", img)
        cv2.setMouseCallback('OriginalPerspective', mouse_click_left)
        
                
    def Btn_4_1_function(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = np.copy(img_gray)
        x, y = np.mgrid[-1:2, -1:2]
        
        a = 2
        gaussian_array = np.exp(-(x**2+y**2)/(2*a*a))/(2*3.1415926*a*a)
        gaussian_sum = np.sum(np.sum(gaussian_array))
        
        for i in range(1,img_gray.shape[0]-1):
            for j in range(1,img_gray.shape[1]-1):
                temp = 0
                for k in range(0,3):
                    for m in range(0,3):
                        temp  = temp + (gaussian_array[k][m]*img_gray[i+k-1][j+m-1])
                gaussian[i][j] = temp/gaussian_sum

        cv2.imshow("Gray", img_gray)
        cv2.imshow("Gaussian", gaussian)
        
    def Btn_4_2_function(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = np.copy(img_gray)
        x, y = np.mgrid[-1:2, -1:2]
        
        a = 2
        gaussian_array = np.exp(-(x*x+y*y)/(2*a*a))/(2*math.pi*a*a)
        
        for i in range(1,img_gray.shape[0]-1):
            for j in range(1,img_gray.shape[1]-1):
                temp = 0
                for k in range(0,3):
                    for m in range(0,3):
                        temp  = temp + (gaussian_array[k][m]*img_gray[i+k-1][j+m-1])
                gaussian[i][j] = temp
                
        
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        
        Sobel_x = np.copy(gaussian)

        for i in range(1,gaussian.shape[0]-1):
            for j in range(1,gaussian.shape[1]-1):
                Sobel_x[i][j] = np.sum(Gx*gaussian[i-1:i+2, j-1:j+2])
                Sobel_x[i][j] = math.sqrt(Sobel_x[i][j]*Sobel_x[i][j])
                
        pdf = np.zeros(256)
        cdf = np.zeros(256)
        h = np.zeros(256)


        """set probability density function"""
        for i in range(0,Sobel_x.shape[0]):
            for j in range(0,Sobel_x.shape[1]):
                pdf[Sobel_x[i][j]] = pdf[Sobel_x[i][j]] + 1
        
        """set Cumulative Distribution Function"""
        cdf[0] = pdf[0]
        for i in range(1,256):
            cdf[i] = pdf[i] + cdf[i-1]

        """Histogram equalization calculation"""
        cdfmin = min(cdf)
        for i in range(0,256):
            h[i] = round((cdf[i]-cdfmin) / ((Sobel_x.shape[0]*Sobel_x.shape[1]) - cdfmin) * 255)

        """set new array after Histogram equalization"""
        equalization = np.copy(Sobel_x)
        
        for i in range(0,Sobel_x.shape[0]):
            for j in range(0,Sobel_x.shape[1]):
                equalization[i][j] = h[Sobel_x[i][j]]
                
        _, equalization = cv2.threshold(equalization, 200, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Sobel_x", equalization)
        

    def Btn_4_3_function(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = np.copy(img_gray)
        x, y = np.mgrid[-1:2, -1:2]
        
        a = 2
        gaussian_array = np.exp(-(x*x+y*y)/(2*a*a))/(2*math.pi*a*a)
        
        for i in range(1,img_gray.shape[0]-1):
            for j in range(1,img_gray.shape[1]-1):
                temp = 0
                for k in range(0,3):
                    for m in range(0,3):
                        temp  = temp + (gaussian_array[k][m]*img_gray[i+k-1][j+m-1])
                gaussian[i][j] = temp
                
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        
        Sobel_y = np.copy(gaussian)
        for i in range(1,gaussian.shape[0]-1):
            for j in range(1,gaussian.shape[1]-1):
                Sobel_y[i][j] = np.sum(Gy*gaussian[i-1:i+2, j-1:j+2])
                Sobel_y[i][j] = math.sqrt(Sobel_y[i][j]*Sobel_y[i][j])

                
        
        pdf = np.zeros(256)
        cdf = np.zeros(256)
        h = np.zeros(256)


        """set probability density function"""
        for i in range(0,Sobel_y.shape[0]):
            for j in range(0,Sobel_y.shape[1]):
                pdf[Sobel_y[i][j]] = pdf[Sobel_y[i][j]] + 1
        
        """set Cumulative Distribution Function"""
        cdf[0] = pdf[0]
        for i in range(1,256):
            cdf[i] = pdf[i] + cdf[i-1]

        """Histogram equalization calculation"""
        cdfmin = min(cdf)
        for i in range(0,256):
            h[i] = round((cdf[i]-cdfmin) / ((Sobel_y.shape[0]*Sobel_y.shape[1]) - cdfmin) * 255)

        """set new array after Histogram equalization"""
        equalization = np.copy(Sobel_y)
        
        for i in range(0,Sobel_y.shape[0]):
            for j in range(0,Sobel_y.shape[1]):
                equalization[i][j] = h[Sobel_y[i][j]]
                
        _, equalization = cv2.threshold(equalization, 200, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Sobel_y", equalization)
        
    def Btn_4_4_function(self):
        img = cv2.imread('./images/School.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = np.copy(img_gray)
        x, y = np.mgrid[-1:2, -1:2]
        
        a = 2
        gaussian_array = np.exp(-(x*x+y*y)/(2*a*a))/(2*math.pi*a*a)
        
        for i in range(1,img_gray.shape[0]-1):
            for j in range(1,img_gray.shape[1]-1):
                temp = 0
                for k in range(0,3):
                    for m in range(0,3):
                        temp  = temp + (gaussian_array[k][m]*img_gray[i+k-1][j+m-1])
                gaussian[i][j] = temp
                
        
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        

        Sobel = np.copy(gaussian)
        for i in range(1,gaussian.shape[0]-1):
            for j in range(1,gaussian.shape[1]-1):
                Sobel_x = np.sum(Gx*gaussian[i-1:i+2, j-1:j+2])
                Sobel_y = np.sum(Gy*gaussian[i-1:i+2, j-1:j+2])
                Sobel[i][j] = math.sqrt(Sobel_x*Sobel_x + Sobel_y*Sobel_y)
                
        
        pdf = np.zeros(256)
        cdf = np.zeros(256)
        h = np.zeros(256)


        """set probability density function"""
        for i in range(0,Sobel.shape[0]):
            for j in range(0,Sobel.shape[1]):
                pdf[Sobel[i][j]] = pdf[Sobel[i][j]] + 1
        
        """set Cumulative Distribution Function"""
        cdf[0] = pdf[0]
        for i in range(1,256):
            cdf[i] = pdf[i] + cdf[i-1]

        """Histogram equalization calculation"""
        cdfmin = min(cdf)
        for i in range(0,256):
            h[i] = round((cdf[i]-cdfmin) / ((Sobel.shape[0]*Sobel.shape[1]) - cdfmin) * 255)

        """set new array after Histogram equalization"""
        equalization = np.copy(Sobel)
        
        for i in range(0,Sobel.shape[0]):
            for j in range(0,Sobel.shape[1]):
                equalization[i][j] = h[Sobel[i][j]]
                
        _, equalization = cv2.threshold(equalization, 200, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Sobel", equalization)
        
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
