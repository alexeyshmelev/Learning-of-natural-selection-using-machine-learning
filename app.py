import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QMainWindow, QFileDialog, QStackedLayout, QAction
from PyQt5.QtGui import QIcon, QPalette, QColor, QLinearGradient, QBrush, QFont
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


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initui()
        self.step = 1
        self.file = None

    def initui(self):

        self.resize(720, 460)
        self.setWindowTitle('Kolibri')
        self.setWindowIcon(QIcon('icon.png'))
        stackedLayout = QStackedLayout()
        menuBar = self.menuBar()
        toolsMenu = menuBar.addMenu('Tools')
        helpMenu = menuBar.addMenu('Help')
        aboutMenu = menuBar.addMenu('About')

        toolsAction = QAction('Restart', self)
        toolsAction.setShortcut('Ctrl+z')
        toolsAction.triggered.connect(self.restart)
        toolsMenu.addAction(toolsAction)
        aboutAction = QAction('Authors', self)
        aboutMenu.addAction(aboutAction)

        self.restart()

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
        gradient = QLinearGradient(-100, 600, 1200, -100)
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
            gradient = QLinearGradient(-100, 600, 1200, -100)
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
        model_sls.load_state_dict(torch.load('rnn.pth', map_location=device))
        model_sls.eval()

        m = nn.Softmax(dim=1)
        test_data = []
        for line in self.file:
            array = line.split('\t')
            test_data += [float(array[2])]
        self.file.close()
        test_data = torch.tensor(test_data)
        test_outputs = model_sls(test_data)
        test_outputs = m(test_outputs).cpu().detach().numpy().flatten()
        max = np.argmax(test_outputs)
        result = 'Natural selection was in locus {:.2} with the probability of {}'.format(max/100, test_outputs[max])

        return result


if __name__ == '__main__':

    app = QApplication([])
    application = App()
    sys.exit(app.exec_())