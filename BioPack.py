import sys
import matplotlib.pyplot as plt
import requests
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QMainWindow, QFileDialog, QStackedLayout, QAction, QMessageBox
from PyQt5.QtGui import QIcon, QPalette, QColor, QLinearGradient, QBrush, QFont, QDesktopServices, QPixmap
from PyQt5.QtCore import QUrl
import torch
import numpy as np
import torch.nn as nn


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


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initui()
        self.step = 1
        self.file = None

    def initui(self):

        self.resize(1200, 900)
        self.setWindowTitle('BioPack')
        self.setWindowIcon(QIcon('BioPack.png'))
        stackedLayout = QStackedLayout()
        menuBar = self.menuBar()
        toolsMenu = menuBar.addMenu('Tools')
        helpMenu = menuBar.addMenu('Help')
        aboutMenu = menuBar.addMenu('About')

        toolsAction = QAction('Restart', self)
        toolsAction.setShortcut('Ctrl+z')
        toolsAction.triggered.connect(self.restart)
        toolsMenu.addAction(toolsAction)
        helpAction = QAction('See on GitHub', self)
        helpMenu.addAction(helpAction)
        aboutAction = QAction('Authors', self)
        version = QAction('v1.0.0', self)
        aboutMenu.addAction(aboutAction)
        aboutMenu.addAction(version)

        self.restart()

        response = requests.get('##########')
        response = response.text.split('#')
        if int(response[0]) > 1 or int(response[1]) > 0 or int(response[2]) > 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("New version (" + response[0] + "." + response[1] + "." + response[2] + ") of BioPack available for downloading. Do you want to visit our GitHub to download latest version?")
            msgBox.setWindowTitle("Update")
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            feedback = msgBox.exec()
            if feedback == QMessageBox.Ok:
                QDesktopServices.openUrl(QUrl('https://github.com/Grenlex/Learning-of-natural-selection-using-machine-learning/releases'))

        self.show()

    def restart(self):
        choose_file = QPushButton('Choose file')
        choose_file.setFont(QFont("San-Serif", 17))
        choose_file.setStyleSheet('''border: 2px solid #ffffff; border-radius: 10px; background: rgba(255, 255, 255, 0.8); color: rgb(0, 150, 115); padding-left: 30px; padding-right: 30px; padding-top: 10px; padding-bottom: 10px;''')
        choose_file.clicked.connect(self.changeStep)
        hint = QLabel('Step 1: choose file you want to analyze')
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
        page1.addWidget(choose_file, 0, 0, QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter)
        page1.addWidget(hint, 1, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.setCentralWidget(initial_widget)

        self.step = 1

    def changeStep(self):

        if self.step == 2:
            widget = QWidget()
            widget.setAutoFillBackground(True)
            palette = widget.palette()
            gradient = QLinearGradient(-100, 1000, 1800, -100)
            gradient.setColorAt(0.0, QColor(0, 150, 115))
            gradient.setColorAt(1.0, QColor(255, 255, 255))
            palette.setBrush(QPalette.Window, QBrush(gradient))
            widget.setPalette(palette)
            page2 = QGridLayout(widget)

            result = self.analyze()

            title_result = QLabel('Results:')
            title_result.setFont(QFont("San-Serif", 17, QFont.Bold))
            title_result.setStyleSheet('''color: rgb(255, 255, 255);''')

            result = QLabel(result)
            result.setFont(QFont("San-Serif", 12))
            result.setStyleSheet('''color: rgb(255, 255, 255);''')
            page2.addWidget(title_result, 0, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
            page2.addWidget(result, 1, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            pixmap = QPixmap("likelihood.png")
            pixmap = pixmap.scaled(700, 700, QtCore.Qt.KeepAspectRatio)
            lbl = QLabel()
            lbl.setPixmap(pixmap)
            lbl.setStyleSheet('''padding-bottom: 30px;''')
            page2.addWidget(lbl, 2, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            self.setCentralWidget(widget)

        if self.step == 1:
            fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
            if fname:
                self.file = open(fname, 'r')
                self.centralWidget().layout().itemAt(0).widget().setText('Analyze ' + fname.split('/')[-1])
                self.centralWidget().layout().itemAt(1).widget().setText('Step 2: press button above again to analyze selected file')
                self.step = 2
            else:
                self.centralWidget().layout().itemAt(0).widget().setText('Error')
                self.centralWidget().layout().itemAt(1).widget().setText('Something went wrong. Restart your program.')

    def analyze(self):

        device = torch.device('cpu')

        model_sls = SLS(1, 1, 1, 100, 100)  # определяем класс с помощью уже натренированной сети
        model_sls.load_state_dict(torch.load('rnn_v_1_0_0.pth', map_location=device))
        model_sls.eval()

        m = nn.Softmax(dim=1)
        test_data = []
        for line in self.file:
            array = line.split('\t')
            test_data += [float(array[2])]
        test_data = torch.tensor(test_data)
        test_outputs = model_sls(test_data)
        test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
        max = np.argmax(test_outputs)
        result_locus = 'Natural selection was in locus {:.2} with the probability of {:.2}'.format(max/100, test_outputs[max])

        positions = [round(i/100, 2) for i in range(100)]
        plt.clf()
        plt.plot(positions, test_outputs, '.-g', label='Likelihood along all genome', alpha=0.5)
        plt.legend(loc='best')
        plt.savefig('likelihood.png')

        model_gen = GEN(100)  # определяем класс с помощью уже натренированной сети
        model_gen.load_state_dict(torch.load('transformer_gen_v_1_0_0.pth', map_location=device))
        model_gen.eval()

        self.file.seek(0)
        test_data = []
        for line in self.file:
            array = line.split('\t')
            test_data += [float(array[2]), float(array[3]), float(array[4]), float(array[5]), float(array[6])]
        test_data = torch.tensor(test_data).float().view(1, 1, 100 * 5)
        test_outputs = model_gen(test_data).cpu().detach().numpy().flatten().item()
        result_gen = 'We are talking about {}th generation'.format(round(test_outputs))

        model_frc = FRC(100)  # определяем класс с помощью уже натренированной сети
        model_frc.load_state_dict(torch.load('transformer_force_v_1_0_0.pth', map_location=device))
        model_frc.eval()

        test_outputs = model_frc(test_data).cpu().detach().numpy().flatten().item()
        result_frc = 'The force of natural selection was {}'.format(round(test_outputs/100, 2))

        model_adm = ADM(100)  # определяем класс с помощью уже натренированной сети
        model_adm.load_state_dict(torch.load('transformer_adm_v_1_0_0.pth', map_location=device))
        model_adm.eval()

        test_outputs = model_adm(test_data).cpu().detach().numpy().flatten().item()
        result_adm = 'The admixture on the first stage was {}'.format(round(test_outputs / 100, 2))

        result = result_locus + '\n' + result_gen + '\n' + result_frc + '\n' + result_adm

        return result


if __name__ == '__main__':

    app = QApplication([])
    application = App()
    sys.exit(app.exec_())
