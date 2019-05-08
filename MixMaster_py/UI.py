# coding: utf-8

import sys
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from run import DqnProgram

import threading
import matplotlib.pyplot as plt


class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("mainWindow.ui", self)

        self.weight_file = None
        self.info = ['없음','비실행중']
        self.training_result = []
        self.setInfo("없음")
        self.running = None
        self.ui.show()

    def setInfo(self, file=None, msg=None):
        baseText = "현재 로드된 모델 파일 : {}\n마지막 메시지\n: {}"
        self.info[0] = file if not file is None else self.info[0]
        self.info[1] = msg if not msg is None else self.info[1]
        self.ui.modelInfoLabel.setText(baseText.format(*self.info))

    @pyqtSlot()
    def ppoSelect(self):
        print("sel ppo")

    @pyqtSlot()
    def dqnSelect(self):
        print("sel dqn")

    @pyqtSlot()
    def changedTrading(self):
        list = self.ui.tradingList
        print(dir(list))
        for a in list.selectedItems():
            print(a.text())

    @pyqtSlot()
    def changedSubject(self):
        pass


    @pyqtSlot()
    def createModel(self):
        pass

    @pyqtSlot()
    def loadModel(self):
        pass

    @pyqtSlot()
    def saveModel(self):
        pass

    @pyqtSlot()
    def modelTraining(self):
        if not self.running is None:
            return
        self.results, self.training_result = [], []
        self.running = threading.Thread(target=DqnProgram, args=(['-m','train','-i','10000000'],self.setInfo, self.training_result))
        self.running.daemon = True
        self.running.start()


    @pyqtSlot()
    def modelTest(self):
        if not self.running is None:
            return
        self.results, self.training_result = [], []
        self.running = threading.Thread(target=DqnProgram, args=(['-m','test','-i','10000000','-w','weights/201904281053-dqn.h5'],self.setInfo, self.training_result))
        self.running.daemon = True
        self.running.start()

    @pyqtSlot()
    def showGraph(self):
        plt.plot(self.training_result)
        plt.ylabel("hoho")
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())
    