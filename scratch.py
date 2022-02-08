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
    path = r'C:\HSE\EPISTASIS\nn\next_gen_simulation_usa'
    temp_data = []
    inputs = []

    px = 1 / plt.rcParams['figure.dpi']
    plt.clf()
    fig, ax = plt.subplots(10, 5, figsize=(6400 * px, 6400 * px))
    # ax[0][0].set_aspect('auto')
    # ax[0][1].set_aspect('auto')
    # ax[0][2].set_aspect('auto')
    # ax[0][3].set_aspect('auto')
    # ax[0][4].set_aspect('auto')

    file_list = sorted(os.listdir(path))

    for i in range(len(file_list)):
    # for i in range(0, 100000, 1000):
        if (i % 100) == 0:
            print("Train file ", i, flush=True)

        exact_file = path + '/' + file_list[i]
        file = open(exact_file, 'r')
        for line in file:
            array = line.split('\t')
            temp = [array[2], array[3], array[4], array[5], array[6]]
            temp = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp]
            if temp[0] != 0:
                temp[0] = temp[0] * 10
            if temp[1] != 0:
                temp[1] = temp[1] * 1000
            if temp[2] != 0:
                temp[2] = temp[2] * 1000
            if temp[3] != 0:
                temp[3] = temp[3] * 1000000
            if temp[4] != 0:
                temp[4] = temp[4] * 1000000
            temp_data += [temp]
        # temp_data.append(float(exact_file.split('_')[4]))
        temp = np.array(temp_data)
        start = temp[:500, :]
        end = temp[:499:-1, :]
        temp_data = np.add(start, end)

        if file_list[i].split('_')[0] == '100':
            ax[0][0].plot(temp_data[:, 0], alpha=0.02)
            ax[0][1].plot(temp_data[:, 1], alpha=0.02)
            ax[0][2].plot(temp_data[:, 2], alpha=0.02)
            ax[0][3].plot(temp_data[:, 3], alpha=0.02)
            ax[0][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '200':
            ax[1][0].plot(temp_data[:, 0], alpha=0.02)
            ax[1][1].plot(temp_data[:, 1], alpha=0.02)
            ax[1][2].plot(temp_data[:, 2], alpha=0.02)
            ax[1][3].plot(temp_data[:, 3], alpha=0.02)
            ax[1][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '300':
            ax[2][0].plot(temp_data[:, 0], alpha=0.02)
            ax[2][1].plot(temp_data[:, 1], alpha=0.02)
            ax[2][2].plot(temp_data[:, 2], alpha=0.02)
            ax[2][3].plot(temp_data[:, 3], alpha=0.02)
            ax[2][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '400':
            ax[3][0].plot(temp_data[:, 0], alpha=0.02)
            ax[3][1].plot(temp_data[:, 1], alpha=0.02)
            ax[3][2].plot(temp_data[:, 2], alpha=0.02)
            ax[3][3].plot(temp_data[:, 3], alpha=0.02)
            ax[3][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '500':
            ax[4][0].plot(temp_data[:, 0], alpha=0.02)
            ax[4][1].plot(temp_data[:, 1], alpha=0.02)
            ax[4][2].plot(temp_data[:, 2], alpha=0.02)
            ax[4][3].plot(temp_data[:, 3], alpha=0.02)
            ax[4][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '600':
            ax[5][0].plot(temp_data[:, 0], alpha=0.02)
            ax[5][1].plot(temp_data[:, 1], alpha=0.02)
            ax[5][2].plot(temp_data[:, 2], alpha=0.02)
            ax[5][3].plot(temp_data[:, 3], alpha=0.02)
            ax[5][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '700':
            ax[6][0].plot(temp_data[:, 0], alpha=0.02)
            ax[6][1].plot(temp_data[:, 1], alpha=0.02)
            ax[6][2].plot(temp_data[:, 2], alpha=0.02)
            ax[6][3].plot(temp_data[:, 3], alpha=0.02)
            ax[6][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '800':
            ax[7][0].plot(temp_data[:, 0], alpha=0.02)
            ax[7][1].plot(temp_data[:, 1], alpha=0.02)
            ax[7][2].plot(temp_data[:, 2], alpha=0.02)
            ax[7][3].plot(temp_data[:, 3], alpha=0.02)
            ax[7][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '900':
            ax[8][0].plot(temp_data[:, 0], alpha=0.02)
            ax[8][1].plot(temp_data[:, 1], alpha=0.02)
            ax[8][2].plot(temp_data[:, 2], alpha=0.02)
            ax[8][3].plot(temp_data[:, 3], alpha=0.02)
            ax[8][4].plot(temp_data[:, 4], alpha=0.02)
        if file_list[i].split('_')[0] == '1000':
            ax[9][0].plot(temp_data[:, 0], alpha=0.02)
            ax[9][1].plot(temp_data[:, 1], alpha=0.02)
            ax[9][2].plot(temp_data[:, 2], alpha=0.02)
            ax[9][3].plot(temp_data[:, 3], alpha=0.02)
            ax[9][4].plot(temp_data[:, 4], alpha=0.02)


        temp_data = []
        file.close()

    gaps = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    for i in range(10):
        ax[i][0].set_title(f'Frequency ({gaps[i]})', fontsize=8)
        ax[i][1].set_title(f'Mean tract length of ancestry 0 ({gaps[i]})', fontsize=8)
        ax[i][2].set_title(f'Mean tract length of ancestry 1 ({gaps[i]})', fontsize=8)
        ax[i][3].set_title(f'Variance of tract length of ancestry 0 ({gaps[i]})', fontsize=8)
        ax[i][4].set_title(f'Variance of tract length of ancestry 1 ({gaps[i]})', fontsize=8)

    fig.savefig("statistics.png")

    inputs = np.array(inputs)
    print(inputs.shape)

    # plt.clf()
    # plt.plot(test_data[1000, :, 4])
    # plt.show()
    # for step in range(100000):
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
        # with open(r'C:\HSE\EPISTASIS\nn\all_inputs_train.npy', 'rb') as f:
        #     test_data = np.squeeze(np.load(f)[step])

        # test_data[:, :] = test_data[:, :] - np.concatenate((test_data, test_data[-1].reshape((1, 5))))[1:, :]
        # print(np.min(test_data))
        # print(np.max(test_data))

        # print(np.array(test_data).shape)
        #
        # y_label_list = ['Frequency of ancestry 0', 'Mean tract length of ancestry 0',
        #                 'Mean tract length of ancestry 1', 'Variance in tract length of ancestry 0',
        #                 'Variance in tract length of ancestry 1']
        # fig, (ax0, ax) = plt.subplots(1, 2)
        # ax0.axis('off')
        # img = ax.imshow(np.array(test_data).T)
        # x_left, x_right = ax.get_xlim()
        # y_low, y_high = ax.get_ylim()
        # ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 1)
        # ax.set_yticks([0, 1, 2, 3, 4])
        # ax.set_yticklabels(y_label_list)
        # fig.colorbar(img)
        # plt.show()

        # print(np.array(test_data).reshape((5, 1000))[:, 412:420])
