import concurrent
import gc
import os
import sys

import cv2
import numpy
import torch

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi

from sklearn.decomposition import PCA


class Mywindowpanel(QMainWindow):

    def __init__(self):
        super(Mywindowpanel, self).__init__()

        self.count_true = 0
        self.count_fail = 0

        self.link_folder_images = None
        self.link_model = None

        self.list_images_fail = []
        self.count_display_images_fail = 0


        self.model = None

        loadUi('evaluate_model.ui', self)

        # self.bt_start.clicked.connect(self.button_start)
        self.bt_detection.clicked.connect(self.load_images)
        self.bt_evaluate_model.clicked.connect(self.button_evaluate_model)
        self.bt_browser_model.clicked.connect(self.button_browser_model)
        self.bt_browser_images.clicked.connect(self.button_browser_images)
        self.bt_next_image.clicked.connect(self.button_next_image)
        self.bt_previous_image.clicked.connect(self.button_previous_image)

    @pyqtSlot()
    def button_browser_images(self):
        # self.link_folder_images = 'E:/vinh_project_nissin/vinh_dataset/coco128_roller_back/images/train'
        self.link_folder_images = QFileDialog.getExistingDirectory()

        self.line_edit_images.setText(self.link_folder_images)
        self.memory()
    @pyqtSlot()
    def button_browser_model(self):
        self.link_model = QFileDialog.getOpenFileName(filter='*.pt *.h5')
        self.model = torch.hub.load('E:/vinh_project_nissin/project_rocket_arm/pytorch/yolov5', 'custom',path=self.link_model[0], source='local')

        self.line_edit_model.setText(self.link_model[0])
        #
        # self.link_model = 'E:/vinh_project_nissin/project_rocket_arm/pytorch/dist/demo_roller_back.pt'
        # self.model = torch.hub.load('E:/vinh_project_nissin/project_rocket_arm/pytorch/yolov5', 'custom',path=self.link_model, source='local')
        #
        # self.line_edit_model.setText(self.link_model)

    @pyqtSlot()
    def load_images(self):
        # name_origin = ''
        self.list_images_fail = []
        labels_folder = self.link_folder_images.replace('images', 'labels')

        # for filename_image in os.listdir(self.link_folder_images):
        for filename_image in (filename for filename in os.listdir(self.link_folder_images) if filename.endswith('.jpg')):

            img = cv2.imread(os.path.join(self.link_folder_images, filename_image))
            filename_label = filename_image.replace('.jpg', '.txt')

            name_detection, confidence, image = self.button_detections(img)

            with open(os.path.join(labels_folder, filename_label), 'r') as f:
                data = f.readline().strip().split()
                label = int(data[0])

                # 0 : OK , ng_xuoc : 1 , ng_lechtruc:2 , ng_chuatan:3
                if label == 0:
                    name_origin = 'OK'
                if label == 1:
                    name_origin = 'NG_XUOC'
                if label == 2:
                    name_origin = 'NG_LECHTRUC'
                if label == 3:
                    name_origin = 'NG_CHUATAN'

                # # name_origin = 'OK' if label == 0 else 'NG'
                # if name_detection == 'NG_XUOC' :
                #     pass
                # else:
                #     cv2.imwrite(f'E:/vinh_project_nissin/vinh_dataset/roller_back_3class/images/train/{filename_image}', img)
                # elif name_detection == "NG":
                #     cv2.imwrite(f'E:/vinh_project_nissin/vinh_dataset/roller_back_4class/ng/{filename_image}', img)

                if name_detection == name_origin:
                    self.count_true = self.count_true + 1
                if name_detection != name_origin:
                    self.count_fail = self.count_fail + 1
                    self.list_images_fail.append(filename_image)

        self.lb_check_fail.setText("Done")
        self.lb_evaluate.setText("count_true: " + str(self.count_true) +
                                "\n" + "count_flase: " + str(self.count_fail))
        print(sys.getsizeof(self.load_images), 'bytes')
        print(self.list_images_fail)

    @pyqtSlot()
    def button_evaluate_model(self):
        self.count_true = 0
        self.count_fail = 0
        self.count_display_images_fail = 0
        self.lb_evaluate.setText("count_true: " + str(self.count_true) +
                                "\n" + "count_flase: " + str(self.count_fail))

    @pyqtSlot()
    def button_next_image(self):
        self.count_display_images_fail += 1
        if self.count_display_images_fail < len(self.list_images_fail):
            self.display_image_fail(self.count_display_images_fail)
        else:
            self.count_display_images_fail -= 1

    @pyqtSlot()
    def button_previous_image(self):
        self.count_display_images_fail -= 1
        if self.count_display_images_fail > 0:
            self.display_image_fail(self.count_display_images_fail)
        else:
            self.count_display_images_fail += 1

    def button_detections(self, img):
        img_copy = img.copy()
        img_pca = self.image_pca(img_copy)
        detection = self.model(img_pca)
        rusult_detection = detection.pandas().xyxy[0].to_dict(orient="records")
        rusult_sorted = sorted(rusult_detection, key=lambda x: x['confidence'], reverse=True)
        try:
            result = rusult_sorted[0]
            confidence = round(result['confidence'], 2)
            name = result["name"]
            if confidence > 0.4:
                print(f"{confidence}")
                x1 = int(result["xmin"])
                y1 = int(result["ymin"])
                x2 = int(result["xmax"])
                y2 = int(result["ymax"])

                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_copy, name, (x1 + 3, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
            return name, confidence, img_copy
        except:
            return None, None, img_copy
        # cv2.imshow('frame', img)
        # cv2.waitKey(0)

    def button_detections_origin(self, img,name_file_labels):
        img_copy = img.copy()
        labels_folder = f"{self.link_folder_images.replace('images', 'labels')}/{name_file_labels}"
        labels_file = labels_folder.replace('.jpg','.txt')
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        img_h, img_w, _ = img.shape

        for line in lines:
            label = line.split()
            name = 'OK' if label[0] == '0' else 'NG_XUOC' if label[0] == '1' else 'NG_LECHTRUC' if label[0] == '2' else 'NG_CHUATAN' if label[0] == '3' else None
            # name = 'OK' if label[0] == '0' else 'NG_LECHTRUC' if label[0] == '1' else 'NG_CHUATAN' if label[0] == '2' else None

            x_center = float(label[1])
            y_center = float(label[2])
            box_width = float(label[3])
            box_height = float(label[4])

            # Convert from YOLOv5 format to x_min, y_min, x_max, y_max
            x_min = int((x_center - box_width / 2) * img_w)
            y_min = int((y_center - box_height / 2) * img_h)
            x_max = int((x_center + box_width / 2) * img_w)
            y_max = int((y_center + box_height / 2) * img_h)

            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(img_copy, name, (x_min + 3, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            return img_copy

    def image_pca(self, image):
        # Load input image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reshape image matrix into a vector
        img_vec = gray.reshape(-1)

        # Perform PCA on image vector
        pca = PCA(n_components=1)
        pca.fit(img_vec.reshape(-1, 1))
        img_pca = pca.transform(img_vec.reshape(-1, 1))

        # Project image onto eigenvectors
        img_projected = pca.inverse_transform(img_pca)
        # Reshape image vector back into a matrix
        img_reconstructed = img_projected.reshape(gray.shape)

        img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cv2.imshow('PCA Feature Image', img_reconstructed)
        # cv2.imshow('origin', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img_reconstructed

    def display_image_fail(self, number):
        image_origin = cv2.imread(f'{self.link_folder_images}/{self.list_images_fail[number]}')

        # for image in self.list_images_fail:
        name, confidence,image_detection_display = self.button_detections(image_origin)
        image_origin_display = self.button_detections_origin(image_origin, self.list_images_fail[number])

        self.lb_check_fail.setText(f"name_image: {self.list_images_fail[number]} \n confidence_detection: {confidence} \n position: {number}")

        self.display_image(image_origin_display, 1)
        self.display_image(image_detection_display, 2)

    def display_image(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        scale_percent = 100  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim=(width, height)
        # resize image
        imgrezise = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        outimg = QImage(imgrezise, imgrezise.shape[1], imgrezise.shape[0], imgrezise.strides[0], qformat)
        # BGR-->RGB
        outimg = outimg.rgbSwapped()

        if window == 1:
            self.lb_img_origin.setPixmap(QPixmap.fromImage(outimg))
            self.lb_img_origin.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter)
        if window == 2:
            self.lb_detection.setPixmap(QPixmap.fromImage(outimg))
            self.lb_detection.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter)
        if window == 3:
            self.lb_evaluate.setPixmap(QPixmap.fromImage(outimg))
            self.lb_evaluate.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter)

    def memory(self):
        del self.list_images_fail
        gc.collect()
        print('memory')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Mywindowpanel()
    window.setWindowTitle('Mainwindow')
    window.show()
    sys.exit(app.exec())
