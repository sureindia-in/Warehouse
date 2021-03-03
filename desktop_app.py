import cv2
import os
#import base64
from PIL import Image
import PIL
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QDate
from PyQt5.QtGui import QImage, QPixmap
from Pallet_Analysis import *
from palletsnboxes.predictor_yolo_detector.detector_test import Detector
from palletsnboxes.com_ineuron_utils.utils import decodeImage

import datetime
import sys
sys.path.insert(0, './palletsnboxes')

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')



class App(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(App,self).__init__()
        self.setupUi(self)
        #self.label_image_size = (self.Photo.geometry().width(), self.Photo.geometry().height())
        self.filename = "inputImage.jpg"
        self.objectDetection = Detector(self.filename)



        #Update Time and Date
        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")

        self.Date_label.setText(current_date)
        self.Time_label.setText(current_time)

        # button function
        self.UploadImage.clicked.connect(self.Upload_Image_fun)
        self.Predict.clicked.connect(self.Predict_fun)





    # def decodeImage_1(imgstring):
    #     imgdata = base64.b64decode(imgstring)
    #     return imgdata

    def Upload_Image_fun(self):
        file_1,_ = QFileDialog.getOpenFileName(self, 'Open image', '', "Images(*.jpg , *.jpeg, *.png)")
        self.imglabel.setPixmap(QPixmap(file_1))
        im1= Image.open(file_1)
        im1.save("./palletsnboxes/predictor_yolo_detector/inference/images/input.jpg")

        # with open("./palletsnboxes/predictor_yolo_detector/inference/images/" + file_1, 'wb') as f:
        #     f.write(file_1)
        #     f.close()        self.objectDetection = Detector(file_1)
        # App.Save_Uploaded_Image(self.file_1)
        # return file_1

    # def Save_Uploaded_Image(self,image):
    #     self.file_2 = self.image
    #     return self.file_2

    def Predict_fun(self):
        # self.file_3 = self.Save_Uploaded_Image()
        #self.objectDetection = Detector()
        result = myWin.objectDetection.detect()
        # self.Pallets_label.setText(myWin.objectDetection.test1)
        # print(result)
        result_str = list(result)
        # print(result_str)
        # self.Boxes_label.setText(myWin.objectDetection.test2)
        #final_image = App.decodeImage_1(result[0]['image'])
        self.imglabel.setPixmap(QPixmap('Output_images/Good_Predictions/output.jpg'))

        self.Pallets_label.setText(result_str[1])
        self.Boxes_label.setText(result_str[0])





if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = App()
    myWin.show()
    sys.exit(app.exec_())