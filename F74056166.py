# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:01:39 2019

@author: 方嘉祥
"""

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import random

loss_1 = []
loss_50 = []
acc_50 = []
validate_list = []


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
        for i in range(0, 10):
            image_num = random.randint(0, len(x_train))
            #image_num = random.randint(0, len(x_test))
            first_train_img = np.reshape(x_train[image_num, :], (28, 28))
            #first_train_img = np.reshape(x_test[image_num, :], (28, 28))
            plt.matshow(first_train_img, cmap = plt.get_cmap('gray'))
            plt.show()
            print(y_train[image_num])

    def Btn_5_2_function(self):
        print("hyerparameters:")
        print("batch size: 256")
        print("learning rate: 0.001")
        print("optimizer: Adam")
        
    def Btn_5_3_function(self):
        trainer.train_loop(model, train_loader)
        trainer.test(model, test_loader)
        
        plot_range = np.arange(0, len(loss_1), 1)
        plt.plot(plot_range, loss_1[:])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('epoch [0/50]')
        #plt.savefig('Btn_5_3.png')
        plt.show()
        
        print(validate_list)
        
    def Btn_5_4_function(self):
        img = cv2.imread('./Btn_5_4.png')
        cv2.imshow("acc and loss", img)
        
    def Btn_5_5_function(self):
        user_test_num = int(self.Image_index_box.text())
        print(type(y_test))
        print([y_test[user_test_num]])
        user_test_x, user_test_y = np.array([x_test[user_test_num]]), np.array([y_test[user_test_num]])
        user_x_test, user_y_test = [
           torch.from_numpy(user_test_x.reshape(-1, 1, 28, 28)).float(),
           torch.from_numpy(user_test_y.astype('long'))
           ]
        user_test_dataset = TensorDataset(user_x_test, user_y_test)
        user_test_loader = DataLoader(dataset=user_test_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
        
        model = model = torch.load('./F74056166_model.pth')
        model.eval()
        
        
        user_test_imgae = np.reshape(x_test[user_test_num, :], (28, 28))
        plt.matshow(user_test_imgae, cmap = plt.get_cmap('gray'))
        plt.show()

        for text_time in range(10):
            trainer.test(model, user_test_loader)
        
        plt.hist(validate_list, bins=np.linspace(0, 10),  facecolor="blue", edgecolor="black", alpha=0.7)
       # print(validate_list)
        plt.show()
        
        validate_list.clear()
        
        
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#print(x_train[0:10], x_test[0:10])
#print(y_train[0:10], y_test[0:10])

""""""
class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features
   
EPOCHS = 1
BATCH_SIZE = 256
PRINT_FREQ = 100
TRAIN_NUMS = 49000
CUDA = False
device = torch.device("cpu")
kwargs = {}

train_x, train_y = torch.from_numpy(x_train.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y_train.astype('long'))
test_x, test_y = [
   torch.from_numpy(x_test.reshape(-1, 1, 28, 28)).float(),
   torch.from_numpy(y_test.astype('long'))
   ]

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE, **kwargs)
model = LeNet5()

class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.device = device
        
    def train_loop(self, model, train_loader):
        for epoch in range(EPOCHS):
            print("---------------- Epoch {} ----------------".format(epoch))
            self._training_step(model, train_loader, epoch)
            
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            self._validate(model, test_loader, 0, state="Testing")
            
    def _training_step(self, model, loader, epoch):
        model.train()
        
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            #N = X.shape[0]
            
            self.optimizer.zero_grad()
            outs = model(X)
            y = y.long()
            loss = self.criterion(outs, y)
            
            if step >= 0 and (step % PRINT_FREQ == 0):
                self._state_logging(outs, y, loss, step, epoch, "Training")
            if epoch == 0:
                loss_1.append(loss)
            loss.backward()
            self.optimizer.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                #N = X.shape[0]

                outs = model(X)
                #print(type(outs))
                #print(outs)
                y = y.long()
                validate_list.append(y)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
            loss_50.append(loss)
            acc_50.append(self._accuracy(outs, y))
                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))

        
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc
    
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(params=model.parameters(),lr=1.25e-2, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
trainer = Trainer(criterion, optimizer, device)

"""
for aag in range(0, 50):
    trainer.train_loop(model, train_loader)
    trainer.test(model, test_loader)



plot_range = np.arange(0, len(acc_50), 1)
plt.subplot(2,1,1)
plt.plot(plot_range, acc_50[:])
plt.xlabel('epoch')
plt.ylabel('%')
plt.title('Accuaccy')

plt.subplot(2,1,2)
plot_range = np.arange(0, len(loss_50), 1)
plt.plot(plot_range, loss_50[:])
plt.xlabel('epoch')
plt.ylabel('loss')

plt.savefig('Btn_5_4.png')
plt.show()

torch.save(model, './F74056166_model.pth')
#torch.save(model.state_dict(), './F74056166_model.pth')
"""

""""""
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

print('Hello')
