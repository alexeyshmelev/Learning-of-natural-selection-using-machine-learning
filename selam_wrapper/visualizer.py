import os
import sys
import math
import matplotlib.pyplot as plt
import requests
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QMainWindow, QFileDialog, QStackedLayout, QAction, QMessageBox, QProgressBar, QScrollArea
from PyQt5.QtGui import QIcon, QPalette, QColor, QLinearGradient, QBrush, QFont, QDesktopServices, QPixmap
from PyQt5.QtCore import QUrl, QTimer, QThread, pyqtSignal
import torch
import numpy as np
import torch.nn as nn
from random import randint

if __name__ == '__main__':

    temp_data = []
    folder = r'C:\HSE\EPISTASIS\nn\next_gen_simulation_usatest'
    file_list = sorted(os.listdir(folder))
    num = 700
    for step in range(num, num+1):
        # generation = float(file_list[step].split('_')[0])
        # admixture = float(file_list[step].split('_')[1])
        # force = float(file_list[step].split('_')[2])
        # print(generation, admixture, force)
        # test_data = []
        # exact_file = folder + '/' + file_list[step]
        # file = open(exact_file, 'r')
        # for line in file:
        #     array = line.split('\t')
        #     temp_data += [array[2], array[3], array[4], array[5], array[6]]
        #     temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(1000) for i in temp_data]
        #     if temp_data[0] != 0:
        #         temp_data[0] = -(np.log(temp_data[0] * 1 + 0.1) / np.log(1.01))
        #     if temp_data[1] != 0:
        #         temp_data[1] = -(np.log(temp_data[1] * 100 + 0.1) / np.log(1.01))
        #     if temp_data[2] != 0:
        #         temp_data[2] = -(np.log(temp_data[2] * 100 + 0.1) / np.log(1.01))
        #     if temp_data[3] != 0:
        #         temp_data[3] = -(np.log(temp_data[3] * 1000000 + 0.1) / np.log(1.01))
        #     if temp_data[4] != 0:
        #         temp_data[4] = -(np.log(temp_data[4] * 1000000 + 0.1) / np.log(1.01))
        #     test_data += [temp_data]
        #     temp_data = []
        # file.close()
        with open(r'C:\HSE\EPISTASIS\App\all_inputs_train_10.npy', 'rb') as f:
            test_data = np.squeeze(np.load(f)[step])

        print(np.array(test_data).shape)

        y_label_list = ['Frequency of ancestry 0', 'Mean tract length of ancestry 0',
                        'Mean tract length of ancestry 1', 'Variance in tract length of ancestry 0',
                        'Variance in tract length of ancestry 1']
        fig, (ax0, ax) = plt.subplots(1, 2)
        ax0.axis('off')
        img = ax.imshow(np.array(test_data).T)
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 1)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(y_label_list)
        fig.colorbar(img)
        plt.show()

        # print(np.array(test_data).reshape((5, 1000))[:, 412:420])
