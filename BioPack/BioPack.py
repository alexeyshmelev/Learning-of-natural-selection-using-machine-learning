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


class SLS(nn.Module):

    def __init__(self, sls_input_size, sls_hidden_size, sls_num_layers, sls_sequence_length, sls_num_classes):
        super(SLS, self).__init__()  # вызываем конструктор класса nn.Module через метод super(), здесь показан пример для версии Python 2.0

        self.sls_input_size = sls_input_size
        self.sls_hidden_size = sls_hidden_size
        self.sls_num_layers = sls_num_layers
        self.sls_sequence_length = sls_sequence_length
        self.sls_num_classes = sls_num_classes

        self.rnn = nn.LSTM(input_size=self.sls_input_size, hidden_size=self.sls_hidden_size, batch_first=True,
                           bidirectional=True)
        self.linear = nn.Linear(2 * self.sls_input_size, self.sls_input_size)

    def forward(self, input):
        input = input.view(-1, self.sls_sequence_length,
                           self.sls_input_size)  # (batch_size, sequence_length, input_size)
        output, _ = self.rnn(input)
        output = output.view(self.sls_sequence_length, 2 * self.sls_input_size)
        out = self.linear(output)
        out = out.view(-1, self.sls_num_classes)

        return out


class GEN(nn.Module):

    def __init__(self, gen_num_classes):
        super(GEN, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=gen_num_classes*5, nhead=5, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.decoder = nn.Linear(gen_num_classes*5, 1)

    def forward(self, x):
        output = self.transformer_encoder(x)
        output = self.decoder(output).view(1, 1)
        return output


class FRC(nn.Module):

    def __init__(self, gen_num_classes):
        super(FRC, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=gen_num_classes*5, nhead=5, dim_feedforward=10000, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.decoder = nn.Linear(gen_num_classes*5, 1)

    def forward(self, x):
        output = self.transformer_encoder(x)
        output = self.decoder(output).view(1, 1)
        return output


class ADM(nn.Module):

    def __init__(self, gen_num_classes):
        super(ADM, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=gen_num_classes*5, nhead=5, dim_feedforward=10000, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.decoder = nn.Linear(gen_num_classes*5, 1)

    def forward(self, x):
        output = self.transformer_encoder(x)
        output = self.decoder(output).view(1, 1)
        return output


class INT(nn.Module):
    # def __init__(self):
    #     super(INT, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 128, kernel_size=(5, 1), stride=1, padding=(2, 0))  # (1000, 5)
    #     self.mp1 = nn.MaxPool2d(kernel_size=(5, 1), padding=0)  # (200, 5)
    #     self.conv2 = nn.Conv2d(128, 64, kernel_size=(5, 1), stride=1, padding=(2, 0))  # (200, 5)
    #     self.mp2 = nn.MaxPool2d(kernel_size=(5, 1), padding=0)  # (40, 5)
    #     self.conv3 = nn.Conv2d(64, 32, kernel_size=(5, 1), stride=1, padding=(2, 0))  # (40, 5)
    #     self.mp3 = nn.MaxPool2d(kernel_size=(5, 1), padding=0)  # (8, 5)
    #     self.dropout = nn.Dropout()
    #     self.fc1 = nn.Linear(1280, 640)
    #     self.fc2 = nn.Linear(640, 320)
    #     self.fc3 = nn.Linear(320, 160)
    #     self.fc4 = nn.Linear(160, 30)
    #     self.relu = nn.ReLU()
    #     self.sm = nn.Softmax(dim=1)

    # def forward(self, src):
    #     output = self.relu(self.conv1(src))
    #     output = self.mp1(output)
    #     output = self.relu(self.conv2(output))
    #     output = self.mp2(output)
    #     output = self.relu(self.conv3(output))
    #     output = self.mp3(output)
    #     output = self.dropout(output)
    #     output = self.fc1(output.view(1, 1280))
    #     output = self.fc2(output)
    #     output = self.fc3(output)
    #     output = self.fc4(output)
    #
    #     return self.sm(output.view(3, 10)).view(1, 30)

    def __init__(self, d_model, nhead, dim_feedforward=6000, dropout=0.1):
        super(INT, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.decoder_1 = nn.Linear(5, 3000)
        self.decoder_2 = nn.Linear(3000, 1500)
        self.decoder_3 = nn.Linear(1500, 750)
        self.decoder_4 = nn.Linear(750, 100)
        self.decoder_5 = nn.Linear(1000, 1)
        self.decoder_6 = nn.Linear(100, 30)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.tg = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, src, weights):
        src2, attn = self.self_attn(src, src, src)
        if weights == True:
            print('ATTENTION WEIGHTS', attn)
        src = src + self.dropout1(src2)
        # src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        # src = self.norm2(src)
        output = self.decoder_1(src.view(1000, 5))
        output = self.decoder_2(output.view(1000, 3000))
        output = self.decoder_3(output.view(1000, 1500))
        output = self.decoder_4(output.view(1000, 750))
        output = self.decoder_5(output.view(100, 1000))
        output = self.decoder_6(output.view(1, 100)).view(1, 30)
        # output = self.tg(output)

        return output


class TOF(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2000, dropout=0):
        super(INT, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.decoder_1 = nn.Linear(5, 1000)
        self.decoder_2 = nn.Linear(1000, 500)
        self.decoder_3 = nn.Linear(500, 250)
        self.decoder_4 = nn.Linear(250, 100)
        self.decoder_5 = nn.Linear(1000, 1)
        # self.decoder_6 = nn.Linear(10, 1)
        self.tg = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, src, weights):
        src2, attn = self.self_attn(src, src, src)
        if weights == True:
            print('ATTENTION WEIGHTS', attn)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        output = self.decoder_1(src.view(1000, 5))
        output = self.decoder_2(output.view(1000, 1000))
        output = self.decoder_3(output.view(1000, 500))
        output = self.decoder_4(output.view(1000, 250))
        output = self.decoder_5(output.view(1000, 100)).view(1, 1000)
        # output = self.tg(output)
        output = self.decoder_6(output.view(1, 1000)).view(1, 2)
        return output, attn


class AnalyzeTimeGap(QThread):
    count_changed = pyqtSignal(int)

    def __init__(self, folder, callback):
        super().__init__()
        self.folder = folder
        self.result = []
        self.prob_list = []
        self.finished.connect(lambda: callback(self.result, self.prob_list))

    def run(self):
        device = torch.device('cpu')
        # model_int = INT()  # определяем класс с помощью уже натренированной сети
        model_int = INT(5, 5)
        model_int.load_state_dict(torch.load('transformer_inf_v_1_0_0_THE_LAST_BEST.pth', map_location=device))
        # model_int.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'transformer_inf_v_1_0_0.pth'), map_location=device))
        model_int.eval()

        # m = nn.Softmax(dim=1)
        m = nn.Sigmoid()

        gen_int = [(i + 1) * 100 for i in range(10)]

        adm_start = 0.0002
        adm_end = 0.01
        l = np.log(adm_end) - np.log(adm_start)
        adm_int = [np.exp(l * (i + 1) / 10 + np.log(adm_start)) for i in range(10)]

        frc_start = 0.0005
        frc_end = 0.02
        l = np.log(frc_end) - np.log(frc_start)
        frc_int = [np.exp(l * (i + 1) / 10 + np.log(frc_start)) for i in range(10)]

        def FormTarget(gen, adm, frc):

            gen_array = np.full(10, 0)
            adm_array = np.full(10, 0)
            frc_array = np.full(10, 0)

            adm_num = 0
            frc_num = 0
            gen_num = 0

            for k, elem in enumerate(gen_int):
                if gen == elem:
                    gen_num = k

            for k, elem in enumerate(adm_int):
                if adm > elem:
                    adm_num = k + 1

            for k, elem in enumerate(frc_int):
                if frc > elem:
                    frc_num = k + 1

            gen_array[gen_num] = 1
            adm_array[adm_num] = 1
            frc_array[frc_num] = 1
            arr = np.concatenate([gen_array, adm_array, frc_array]).reshape((3, 10))

            return arr

        count = 0
        data_list_generation = []
        data_list_difference = []
        temp_data = []
        file_list = sorted(os.listdir(self.folder))

        for step in range(len(os.listdir(self.folder))):
            generation = float(file_list[step].split('_')[0])
            admixture = float(file_list[step].split('_')[1])
            force = float(file_list[step].split('_')[2])
            test_data = []
            exact_file = self.folder + '/' + file_list[step]
            file = open(exact_file, 'r')
            for line in file:
                array = line.split('\t')
                temp_data += [array[2], array[3], array[4], array[5], array[6]]
                temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp_data]
                temp_data[0] *= 10
                temp_data[1] *= 1000
                temp_data[2] *= 1000
                temp_data[3] *= 10000000
                temp_data[4] *= 1000000
                test_data += temp_data
                temp_data = []
            file.close()
            # test_data = torch.tensor(test_data).float().view(1, 1, 1000, 5)
            test_data = torch.tensor(test_data).float().view(1000, 1, 5)
            test_outputs = model_int(test_data, False).view(3, 10)
            # test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
            # max = np.argmax(test_outputs)
            # test_outputs = np.sum(test_outputs.cpu().detach().numpy(), axis=1)
            test_outputs = test_outputs.cpu().detach().numpy()
            self.prob_list += [test_outputs]
            targets = FormTarget(generation, admixture, force)
            gen_class = np.argmax(targets[0])
            adm_class = np.argmax(targets[1])
            frc_class = np.argmax(targets[2])
            self.result += ['Generation: {}, Admixture: {}, Force: {}'.format(generation, admixture, force) + '\n' + 'Initial classes: generation - {}, admixture - {}, force - {}'.format(gen_class, adm_class, frc_class) + '\n']
            count = int(step / len(os.listdir(self.folder)) * 100)
            self.count_changed.emit(count)


class Existence(QThread):
    count_changed = pyqtSignal(int)

    def __init__(self, folder, callback):
        super().__init__()
        self.folder = folder
        self.result = []
        self.attn_list = []
        self.finished.connect(lambda: callback(self.result, self.attn_list))

    def run(self):
        answer = ''
        device = torch.device('cpu')
        model_tof = TOF(5, 5)  # определяем класс с помощью уже натренированной сети
        model_tof.load_state_dict(torch.load('transformer_tof_v_1_0_0.pth', map_location=device))
        # model_tof.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'transformer_tof_v_1_0_0.pth'), map_location=device))
        model_tof.eval()
        m = nn.Softmax(dim=1)

        count = 0
        wrong_plus = 0
        wrong_minus = 0
        temp_data = []
        file_list = sorted(os.listdir(self.folder))

        for step in range(len(os.listdir(self.folder))):
            generation = float(file_list[step].split('_')[0])
            test_data = []
            exact_file = self.folder + '/' + file_list[step]
            file = open(exact_file, 'r')
            for line in file:
                array = line.split('\t')
                temp_data += [array[2], array[3], array[4], array[5], array[6]]
                temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp_data]
                temp_data[0] *= 10
                temp_data[1] *= 1000
                temp_data[2] *= 1000
                temp_data[3] *= 10000000
                temp_data[4] *= 1000000
                test_data += temp_data
                temp_data = []
            file.close()
            test_data = torch.tensor(test_data).float().view(1000, 1, 5)
            test_outputs, attn = model_tof(test_data, False)
            self.attn_list += [attn.squeeze().cpu().detach().numpy()]
            arg = np.argmax(m(test_outputs).cpu().detach().numpy().flatten())
            prob = max(m(test_outputs).cpu().detach().numpy().flatten())
            if arg == 0:
                answer = 'No'
            if arg == 1:
                answer = 'Yes'
            # if answer == 'Yes' and generation <= 500:
            #     wrong_minus += 1
            # if answer == 'No' and generation > 500:
            #     wrong_plus += 1
            self.result += ['Initial: {}, Existence of natural selection: {} ------ with the probability of {}'.format(file_list[step], answer, round(prob, 4)) + '\n']
            count = int(step / len(os.listdir(self.folder)) * 100)
            self.count_changed.emit(count)
        print('Wrong plus: {}, wrong minus: {}'.format(wrong_plus, wrong_minus))


class Statistics(QThread):
    count_changed = pyqtSignal(int)

    def __init__(self, folder, callback):
        super().__init__()
        self.x = []
        self.y = []
        self.folder = folder
        self.result = []
        self.list_of_files = []
        self.finished.connect(lambda: callback(self.result, self.x, self.y, self.list_of_files))

    def run(self):
        device = torch.device('cpu')

        model_sls = SLS(1, 1, 1, 100, 100)  # определяем класс с помощью уже натренированной сети
        model_sls.load_state_dict(torch.load('rnn_v_1_0_0.pth', map_location=device))
        # model_sls.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'rnn_v_1_0_0.pth'), map_location=device))
        model_sls.eval()

        model_gen = GEN(100)  # определяем класс с помощью уже натренированной сети
        model_gen.load_state_dict(torch.load('transformer_gen_v_1_0_0.pth', map_location=device))
        # model_gen.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'transformer_gen_v_1_0_0.pth'), map_location=device))
        model_gen.eval()

        model_frc = FRC(100)  # определяем класс с помощью уже натренированной сети
        model_frc.load_state_dict(torch.load('transformer_force_v_1_0_0.pth', map_location=device))
        # model_frc.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'transformer_force_v_1_0_0.pth'), map_location=device))
        model_frc.eval()

        model_adm = ADM(100)  # определяем класс с помощью уже натренированной сети
        model_adm.load_state_dict(torch.load('transformer_adm_v_1_0_0.pth', map_location=device))
        # model_adm.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'transformer_adm_v_1_0_0.pth'), map_location=device))
        model_adm.eval()

        m = nn.Softmax(dim=1)
        file_list = sorted(os.listdir(self.folder))
        count = 0
        self.list_of_files = os.listdir(self.folder)

        for step in range(len(os.listdir(self.folder))):

            exact_file = self.folder + '/' + file_list[step]
            file = open(exact_file, 'r')

            test_data = []
            for line in file:
                array = line.split('\t')
                test_data += [float(array[2])]
            test_data = torch.tensor(test_data)
            test_outputs = model_sls(test_data)
            test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
            max = np.argmax(test_outputs)
            result_locus = 'Natural selection was in locus {:.2} with the probability of {:.2}'.format(max / 100, test_outputs[max])

            positions = [round(i / 100, 2) for i in range(100)]
            self.x += [positions]
            self.y += [test_outputs]

            file.seek(0)
            test_data = []
            for line in file:
                array = line.split('\t')
                test_data += [float(array[2]), float(array[3]), float(array[4]), float(array[5]), float(array[6])]
            test_data = torch.tensor(test_data).float().view(1, 1, 100 * 5)
            test_outputs = model_gen(test_data).cpu().detach().numpy().flatten().item()
            result_gen = 'We are talking about {}th generation'.format(round(test_outputs))

            test_outputs = model_frc(test_data).cpu().detach().numpy().flatten().item()
            result_frc = 'The force of natural selection was {}'.format(round(test_outputs / 100, 2))

            test_outputs = model_adm(test_data).cpu().detach().numpy().flatten().item()
            result_adm = 'The admixture on the first stage was {}'.format(round(test_outputs / 100, 2))

            result = result_locus + '\n' + result_gen + '\n' + result_frc + '\n' + result_adm
            self.result += [result]
            file.close()

            count = int(step / len(os.listdir(self.folder)) * 100)
            self.count_changed.emit(count)


class Visualization(QThread):
    count_changed = pyqtSignal(int)

    def __init__(self, folder, callback):
        super().__init__()
        self.folder = folder
        self.result = []
        self.data = []
        self.finished.connect(lambda: callback(self.result, self.data))

    def run(self):

        gen_int = [(i + 1) * 100 for i in range(10)]

        adm_start = 0.0002
        adm_end = 0.01
        l = np.log(adm_end) - np.log(adm_start)
        adm_int = [np.exp(l * (i + 1) / 10 + np.log(adm_start)) for i in range(10)]

        frc_start = 0.0005
        frc_end = 0.02
        l = np.log(frc_end) - np.log(frc_start)
        frc_int = [np.exp(l * (i + 1) / 10 + np.log(frc_start)) for i in range(10)]

        def FormTarget(gen, adm, frc):

            gen_array = np.full(10, 0)
            adm_array = np.full(10, 0)
            frc_array = np.full(10, 0)

            adm_num = 0
            frc_num = 0
            gen_num = 0

            for k, elem in enumerate(gen_int):
                if gen == elem:
                    gen_num = k

            for k, elem in enumerate(adm_int):
                if adm > elem:
                    adm_num = k + 1

            for k, elem in enumerate(frc_int):
                if frc > elem:
                    frc_num = k + 1

            gen_array[gen_num] = 1
            adm_array[adm_num] = 1
            frc_array[frc_num] = 1
            arr = np.concatenate([gen_array, adm_array, frc_array]).reshape((3, 10))

            return arr

        count = 0
        data_list_generation = []
        data_list_difference = []
        temp_data = []
        file_list = sorted(os.listdir(self.folder))

        for step in range(len(os.listdir(self.folder))):
            generation = float(file_list[step].split('_')[0])
            admixture = float(file_list[step].split('_')[1])
            force = float(file_list[step].split('_')[2])
            test_data = []
            exact_file = self.folder + '/' + file_list[step]
            file = open(exact_file, 'r')
            for line in file:
                array = line.split('\t')
                temp_data += [array[2], array[3], array[4], array[5], array[6]]
                temp_data = [float(i) if i != '-nan' and i != '-nan\n' else float(0) for i in temp_data]
                temp_data[0] *= 10
                temp_data[1] *= 1000
                temp_data[2] *= 1000
                temp_data[3] *= 10000000
                temp_data[4] *= 1000000
                test_data += temp_data
                temp_data = []
            file.close()

            self.data += [test_data]

            targets = FormTarget(generation, admixture, force)
            gen_class = np.argmax(targets[0])
            adm_class = np.argmax(targets[1])
            frc_class = np.argmax(targets[2])
            self.result += ['Generation: {}, Admixture: {}, Force: {}'.format(generation, admixture, force) + '\n' + 'Initial classes: generation - {}, admixture - {}, force - {}'.format(gen_class, adm_class, frc_class) + '\n']
            count = int(step / len(os.listdir(self.folder)) * 100)
            self.count_changed.emit(count)

        self.data = np.array(self.data).reshape((len(os.listdir(self.folder)), 5, 1000))


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.step = 1
        self.folder = None
        self.statistics = None
        self.time_gap = None
        self.tof = None
        self.vis = None
        self.thread = None
        self.progress = None
        self.initui()

    def initui(self):

        self.resize(1200, 900)
        self.setWindowTitle('BioPack v2.1.0')
        # self.setWindowIcon(QIcon(os.path.join(sys._MEIPASS, 'BioPack.png')))
        #self.setWindowIcon(QIcon('BioPack.png'))
        #stackedLayout = QStackedLayout()
        menuBar = self.menuBar()
        toolsMenu = menuBar.addMenu('Tools')
        modeMenu = menuBar.addMenu('Mode')
        helpMenu = menuBar.addMenu('Help')

        toolsAction = QAction('Restart', self)
        toolsAction.setShortcut('Ctrl+z')
        toolsAction.triggered.connect(self.restart)
        toolsMenu.addAction(toolsAction)
        helpAction = QAction('See on GitHub', self)
        helpMenu.addAction(helpAction)
        helpAction.triggered.connect(self.show_git_help)
        statistics = QAction('Statistics', self)
        statistics.triggered.connect(self.set_statistics)
        modeMenu.addAction(statistics)
        time_gap = QAction('Time gap', self)
        time_gap.triggered.connect(self.set_time_gap)
        modeMenu.addAction(time_gap)
        tof = QAction('Existence', self)
        tof.triggered.connect(self.set_tof)
        modeMenu.addAction(tof)
        vis = QAction('Data visualization', self)
        vis.triggered.connect(self.set_vis)
        modeMenu.addAction(vis)

        self.statistics = True
        self.time_gap = False
        self.tof = False
        self.vis = False
        self.restart()

        response = requests.get('http://grenlex.com/bioinformatics/biopack_version.php')
        response = response.text.split('.')
        if int(response[0]) > 2 or int(response[1]) > 0 or int(response[2]) > 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("New version (" + response[0] + "." + response[1] + "." + response[2] + ") of BioPack available for downloading. Do you want to visit our GitHub to download latest version?")
            msgBox.setWindowTitle("Update")
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            feedback = msgBox.exec()
            if feedback == QMessageBox.Ok:
                QDesktopServices.openUrl(QUrl('https://github.com/Grenlex/Learning-of-natural-selection-using-machine-learning/releases'))

        self.show()

    def set_statistics(self):
        self.statistics = True
        self.time_gap = False
        self.tof = False
        self.vis = False
        self.centralWidget().layout().itemAt(1).widget().setText('Step 1: choose file you want to analyze (Mode: statistics)')

    def set_time_gap(self):
        self.statistics = False
        self.time_gap = True
        self.tof = False
        self.vis = False
        self.centralWidget().layout().itemAt(1).widget().setText('Step 1: choose file you want to analyze (Mode: time gap)')

    def set_tof(self):
        self.statistics = False
        self.time_gap = False
        self.tof = True
        self.vis = False
        self.centralWidget().layout().itemAt(1).widget().setText('Step 1: choose file you want to analyze (Mode: existence)')

    def set_vis(self):
        self.statistics = False
        self.time_gap = False
        self.tof = False
        self.vis = True
        self.centralWidget().layout().itemAt(1).widget().setText('Step 1: choose file you want to analyze (Mode: data visualization)')

    def show_git_help(self):
        QDesktopServices.openUrl(QUrl('https://github.com/Grenlex/Learning-of-natural-selection-using-machine-learning'))

    def restart(self):
        choose_file = QPushButton('Choose file')
        choose_file.setFont(QFont("San-Serif", 17))
        choose_file.setStyleSheet('''border: 2px solid #ffffff; border-radius: 10px; background: rgba(255, 255, 255, 0.8); color: rgb(0, 150, 115); padding-left: 30px; padding-right: 30px; padding-top: 10px; padding-bottom: 10px;''')
        choose_file.clicked.connect(self.changeStep)
        hint = QLabel()
        if self.statistics:
            hint.setText('Step 1: choose folder you want to analyze (Mode: statistics)')
        if self.time_gap:
            hint.setText('Step 1: choose folder you want to analyze (Mode: time gap)')
        if self.tof:
            hint.setText('Step 1: choose folder you want to analyze (Mode: existence)')
        if self.vis:
            hint.setText('Step 1: choose folder you want to analyze (Mode: data visualization)')
        hint.setFont(QFont("San-Serif", 10))
        hint.setStyleSheet('''color: rgb(255, 255, 255);''')

        initial_widget = QWidget()
        initial_widget.setAutoFillBackground(True)
        palette = initial_widget.palette()
        gradient = QLinearGradient(-100, 1000, 1800, -100)
        gradient.setColorAt(0.0, QColor(0, 150, 115))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        initial_widget.setPalette(palette)
        page1 = QGridLayout(initial_widget)
        page1.addWidget(choose_file, 1, 0, QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter)
        page1.addWidget(hint, 2, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.setCentralWidget(initial_widget)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(400)
        self.progress.setAlignment(QtCore.Qt.AlignCenter)
        self.progress.setMaximum(100)

        self.step = 1

    def changeStep(self):

        if self.step == 2:

            if self.statistics:
                self.centralWidget().layout().itemAt(1).widget().setText('Processing...')
                self.thread = Statistics(self.folder, self.statistics_results)
                self.thread.start()
                self.thread.count_changed.connect(self.onCountChanged)

            if self.time_gap:
                self.centralWidget().layout().itemAt(1).widget().setText('Processing...')
                self.thread = AnalyzeTimeGap(self.folder, self.time_gap_results)
                self.thread.start()
                self.thread.count_changed.connect(self.onCountChanged)

            if self.tof:
                self.centralWidget().layout().itemAt(1).widget().setText('Processing...')
                self.thread = Existence(self.folder, self.tof_results)
                self.thread.start()
                self.thread.count_changed.connect(self.onCountChanged)

            if self.vis:
                self.centralWidget().layout().itemAt(1).widget().setText('Processing...')
                self.thread = Visualization(self.folder, self.draw_data)
                self.thread.start()
                self.thread.count_changed.connect(self.onCountChanged)

        if self.step == 1:
            folder = QFileDialog.getExistingDirectory(self, 'Open directory', '/home')
            if folder:
                self.folder = folder
                self.centralWidget().layout().setRowStretch(0, 10)
                self.centralWidget().layout().setRowStretch(1, 2)
                self.centralWidget().layout().setRowStretch(2, 2)
                self.centralWidget().layout().setRowStretch(3, 10)
                self.centralWidget().layout().itemAt(0).widget().setText('Analyze ' + folder.split('/')[-1])
                self.centralWidget().layout().itemAt(1).widget().setText('Step 2: press button above again to analyze all files in selected folder')
                self.centralWidget().layout().addWidget(self.progress, 3, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
                self.step = 2
            else:
                self.centralWidget().layout().itemAt(0).widget().setText('Error')
                self.centralWidget().layout().itemAt(1).widget().setText('Something went wrong. Restart your program.')

    def onCountChanged(self, value):
        self.progress.setValue(value)

    def draw_data(self, result, data):
        print('nice')
        widget = QWidget()
        widget.setAutoFillBackground(True)
        palette = widget.palette()
        gradient = QLinearGradient(-100, 1000, 1800, -100)
        gradient.setColorAt(0.0, QColor(0, 150, 115))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        widget.setPalette(palette)
        page = QGridLayout(widget)

        title_result = QLabel('Results:')
        title_result.setFont(QFont("San-Serif", 17, QFont.Bold))
        title_result.setStyleSheet('''color: rgb(255, 255, 255);''')

        row = 0

        for i in range(len(data)):
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(data[i])
            ax.set_aspect('auto')
            fig.savefig('visualization' + str(i) + '.png')
            plt.close()

            row += 1
            res = QLabel(result[i])
            res.setFont(QFont("San-Serif", 12))
            res.setStyleSheet('''color: rgb(255, 255, 255);''')
            page.addWidget(res, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            pixmap = QPixmap('visualization' + str(i) + '.png')
            pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
            lbl = QLabel()
            lbl.setStyleSheet('''padding-bottom: 30px;''')
            lbl.setPixmap(pixmap)
            row += 1
            page.addWidget(lbl, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)

        self.setCentralWidget(scroll)

    def tof_results(self, result, attn_list):
        widget = QWidget()
        widget.setAutoFillBackground(True)
        palette = widget.palette()
        gradient = QLinearGradient(-100, 1000, 1800, -100)
        gradient.setColorAt(0.0, QColor(0, 150, 115))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        widget.setPalette(palette)
        page = QGridLayout(widget)

        title_result = QLabel('Results:')
        title_result.setFont(QFont("San-Serif", 17, QFont.Bold))
        title_result.setStyleSheet('''color: rgb(255, 255, 255);''')
        page.addWidget(title_result, 0, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)

        row = 0

        for i in range(len(attn_list)):
            plt.clf()
            plt.imshow(attn_list[i])
            plt.colorbar()
            plt.savefig('attention' + str(i) + '.png')

            row += 1
            res = QLabel(result[i])
            res.setFont(QFont("San-Serif", 12))
            res.setStyleSheet('''color: rgb(255, 255, 255);''')
            page.addWidget(res, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            pixmap = QPixmap('attention' + str(i) + '.png')
            pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
            lbl = QLabel()
            lbl.setStyleSheet('''padding-bottom: 30px;''')
            lbl.setPixmap(pixmap)
            row += 1
            page.addWidget(lbl, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)

        self.setCentralWidget(scroll)

    def time_gap_results(self, result, probs):
        widget = QWidget()
        widget.setAutoFillBackground(True)
        palette = widget.palette()
        gradient = QLinearGradient(-100, 1000, 1800, -100)
        gradient.setColorAt(0.0, QColor(0, 150, 115))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        widget.setPalette(palette)
        page = QGridLayout(widget)

        title_result = QLabel('Results:')
        title_result.setFont(QFont("San-Serif", 17, QFont.Bold))
        title_result.setStyleSheet('''color: rgb(255, 255, 255);''')

        gen_int = [(i + 1) * 100 for i in range(10)]

        adm_start = 0.0002
        adm_end = 0.01
        l = np.log(adm_end) - np.log(adm_start)
        adm_int = [np.exp(l * (i + 1) / 10 + np.log(adm_start)) for i in range(10)]

        frc_start = 0.0005
        frc_end = 0.02
        l = np.log(frc_end) - np.log(frc_start)
        frc_int = [np.exp(l * (i + 1) / 10 + np.log(frc_start)) for i in range(10)]

        def FormTarget(gen, adm, frc):

            gen_array = np.full(10, 0)
            adm_array = np.full(10, 0)
            frc_array = np.full(10, 0)

            adm_num = 0
            frc_num = 0
            gen_num = 0

            for s, elem in enumerate(gen_int):
                if gen == elem:
                    gen_num = s

            for s, elem in enumerate(adm_int):
                if adm > elem:
                    adm_num = s + 1

            for s, elem in enumerate(frc_int):
                if frc > elem:
                    frc_num = s + 1

            gen_array[gen_num] = 1
            adm_array[adm_num] = 1
            frc_array[frc_num] = 1
            arr = np.concatenate([gen_array, adm_array, frc_array]).reshape((3, 10))

            return arr

        row = 6
        y = [1 for i in range(10)]

        for i in range(len(probs)):
            fig, ax = plt.subplots(1, 1)
            img = ax.imshow(probs[i])
            y_label_list = ['generation', 'admixture', 'force']
            x_label_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            ax.set_xticklabels(x_label_list)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(y_label_list)
            # img.xlabel("class")
            fig.colorbar(img)
            plt.savefig('probability' + str(i) + '.png', bbox_inches="tight")
            plt.close()

            row += 1
            gen_class = np.argmax(probs[i][0])
            adm_class = np.argmax(probs[i][1])
            frc_class = np.argmax(probs[i][2])
            res = QLabel(result[i] + 'Predicted classes: generation - {}, admixture - {}, force - {}'.format(gen_class, adm_class, frc_class) + '\n')
            res.setFont(QFont("San-Serif", 12))
            res.setStyleSheet('''color: rgb(255, 255, 255);''')
            page.addWidget(res, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            pixmap = QPixmap('probability' + str(i) + '.png')
            pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
            lbl = QLabel()
            lbl.setStyleSheet('''padding-bottom: 30px;''')
            lbl.setPixmap(pixmap)
            row += 1
            page.addWidget(lbl, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        interval_str_gen = 'Generation:' + '\n'

        for k, data in enumerate(gen_int):
            if k == 0:
                interval_str_gen += 'Interval ' + str(k) + ': from 0 to ' + str(data) + '\n'
            else:
                interval_str_gen += 'Interval ' + str(k) + ': from ' + str(gen_int[k-1]) + ' to ' + str(data) + '\n'

        interval_str_adm = 'Admixture: ' + '\n'

        for k, data in enumerate(adm_int):
            if k == 0:
                interval_str_adm += 'Interval ' + str(k) + ': from ' + str(round(adm_start, 6)) + ' to ' + str(round(data, 6)) + '\n'
            else:
                interval_str_adm += 'Interval ' + str(k) + ': from ' + str(round(adm_int[k-1], 6)) + ' to ' + str(round(data, 6)) + '\n'

        interval_str_frc = 'Force: ' + '\n'

        for k, data in enumerate(frc_int):
            if k == 0:
                interval_str_frc += 'Interval ' + str(k) + ': from ' + str(round(frc_start, 6)) + ' to ' + str(round(data, 6)) + '\n'
            else:
                interval_str_frc += 'Interval ' + str(k) + ': from ' + str(round(frc_int[k-1], 6)) + ' to ' + str(round(data, 6)) + '\n'

        interval_label = QLabel(interval_str_gen)
        interval_label.setFont(QFont("San-Serif", 12))
        interval_label.setStyleSheet('''color: rgb(255, 255, 255);''')
        page.addWidget(interval_label, 1, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        plt.close()
        plt.figure(figsize=(12, 1))
        plt.plot(gen_int, y, '.-g')
        for i in gen_int:
            plt.annotate(str(i), (i, 1), (i+0.01, 1.01))
        plt.axis('off')
        plt.savefig('gen_intervals.png', bbox_inches="tight")
        pixmap = QPixmap('gen_intervals.png')
        pixmap = pixmap.scaled(900, 900, QtCore.Qt.KeepAspectRatio)
        lbl = QLabel()
        lbl.setStyleSheet('''padding-bottom: 30px;''')
        lbl.setPixmap(pixmap)
        page.addWidget(lbl, 2, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        interval_label = QLabel(interval_str_adm)
        interval_label.setFont(QFont("San-Serif", 12))
        interval_label.setStyleSheet('''color: rgb(255, 255, 255);''')
        page.addWidget(interval_label, 3, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        plt.close()
        plt.figure(figsize=(12, 1))
        plt.plot(adm_int, y, '.-g')
        for i in adm_int:
            plt.annotate(str(round(i, 6)), (i, 1), (i, 1.01), rotation=60)
        plt.axis('off')
        plt.savefig('adm_intervals.png', bbox_inches="tight")
        pixmap = QPixmap('adm_intervals.png')
        pixmap = pixmap.scaled(900, 900, QtCore.Qt.KeepAspectRatio)
        lbl = QLabel()
        lbl.setStyleSheet('''padding-bottom: 30px;''')
        lbl.setPixmap(pixmap)
        page.addWidget(lbl, 4, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        interval_label = QLabel(interval_str_frc)
        interval_label.setFont(QFont("San-Serif", 12))
        interval_label.setStyleSheet('''color: rgb(255, 255, 255);''')
        page.addWidget(interval_label, 5, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        plt.close()
        plt.figure(figsize=(12, 1))
        plt.plot(frc_int, y, '.-g')
        for i in frc_int:
            plt.annotate(str(round(i, 6)), (i, 1), (i, 1.01), rotation=60)
        plt.axis('off')
        plt.savefig('frc_intervals.png', bbox_inches="tight")
        pixmap = QPixmap('frc_intervals.png')
        pixmap = pixmap.scaled(900, 900, QtCore.Qt.KeepAspectRatio)
        lbl = QLabel()
        lbl.setStyleSheet('''padding-bottom: 30px;''')
        lbl.setPixmap(pixmap)
        page.addWidget(lbl, 6, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        # result = QLabel(result)
        # result.setFont(QFont("San-Serif", 12))
        # result.setStyleSheet('''color: rgb(255, 255, 255);''')
        page.addWidget(title_result, 0, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        # page.addWidget(result, 1, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        # plt.clf()
        # plt.figure(1)
        # plt.hist(generation, label='Distribution of detected generations')
        # plt.legend(loc='best')
        # plt.savefig('generation.png')
        # plt.figure(2)
        # plt.hist(difference, label='Distribution of difference')
        # plt.axvline(x=np.mean(difference), color='r', label='Mean difference')
        # plt.legend(loc='best')
        # plt.savefig('difference.png')
        #
        # pixmap = QPixmap('generation.png')
        # pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
        # lbl = QLabel()
        # lbl.setStyleSheet('''padding-bottom: 30px;''')
        # lbl.setPixmap(pixmap)
        # page.addWidget(lbl, 2, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        #
        # pixmap = QPixmap('difference.png')
        # pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
        # lbl = QLabel()
        # lbl.setStyleSheet('''padding-bottom: 30px;''')
        # lbl.setPixmap(pixmap)
        # page.addWidget(lbl, 3, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)

        self.setCentralWidget(scroll)

    def statistics_results(self, result, x, y, list_of_files):
        widget = QWidget()
        widget.setAutoFillBackground(True)
        palette = widget.palette()
        gradient = QLinearGradient(-100, 1000, 1800, -100)
        gradient.setColorAt(0.0, QColor(0, 150, 115))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        widget.setPalette(palette)
        page = QGridLayout(widget)

        title_result = QLabel('Results:')
        title_result.setFont(QFont("San-Serif", 17, QFont.Bold))
        title_result.setStyleSheet('''color: rgb(255, 255, 255);''')

        page.addWidget(title_result, 0, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)

        row = 0
        for i in range(len(x)):
            plt.clf()
            plt.plot(x[i], y[i], '.-g', label='Likelihood along all genome', alpha=0.5)
            plt.legend(loc='best')
            plt.savefig('likelihood' + str(i) + '.png')

            res = QLabel(list_of_files[i])
            res.setFont(QFont("San-Serif", 14))
            res.setStyleSheet('''color: rgb(255, 255, 255);''')
            row += 1
            page.addWidget(res, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            res = QLabel(result[i])
            res.setFont(QFont("San-Serif", 12))
            res.setStyleSheet('''color: rgb(255, 255, 255);''')
            row += 1
            page.addWidget(res, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            pixmap = QPixmap('likelihood' + str(i) + '.png')
            pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
            lbl = QLabel()
            lbl.setStyleSheet('''padding-bottom: 30px;''')
            lbl.setPixmap(pixmap)
            row += 1
            page.addWidget(lbl, row, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)

        self.setCentralWidget(scroll)


if __name__ == '__main__':

    app = QApplication([])
    application = App()
    sys.exit(app.exec_())
