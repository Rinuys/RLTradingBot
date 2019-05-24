# coding: utf-8

import sys
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from env.Strategies import *
from run import main as program

import threading
import matplotlib.pyplot as plt



class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("mainWindow.ui", self)

        self.weight_path = None # 현재 선택된 모델이 담긴 폴더 TODO 모델 파일? 폴더?
        self.info = ['없음','비실행중']  # ['현재 모델파일', '학습/테스트 메시지']
        self.training_result = [] # 학습/테스트 스레드가 남길 히스토리배열
        self.setInfo("없음")
        self.running = None
        self.ui.show()

        self.tradingList.addItems([ func['name'] for func in strategies]) # 트레이딩 알고리즘 메뉴 등록
        self.subjectList.addItems(['금','석유']) # 종목 선택 메뉴 등록 TODO Ui에 종목 등록 유언성 필요
        self.viewSubjectList.addItems(['금','석유']) # 지표 선택 메뉴 등록 TODO Ui에 종목 등록 유언성 필요
        self.selected_subject = ['금', ['금']] # 현재 선택된 종목
        self.selected_trading = [] # 현재 선택된 알고리즘
        self.selected_learn = 'ppo' # 현재 선택된 강화학습방식, ppo or dqn

        # 기본값 설정
        self.subjectList.itemAt(0,0).setSelected(True)
        self.viewSubjectList.itemAt(0,0).setSelected(True)

    def setInfo(self, file=None, msg=None):
        # UI의 메시지 업데이트
        baseText = "현재 로드된 모델 파일 : {}\n마지막 메시지\n: {}"
        self.info[0] = file if not file is None else self.info[0]
        self.info[1] = msg if not msg is None else self.info[1]
        self.ui.modelInfoLabel.setText(baseText.format(*self.info))


    #이벤트 슬롯 구현
    @pyqtSlot()
    def ppoSelect(self):
        ''' PPO 선택 이벤트핸들러'''
        self.selected_learn = 'ppo'
        print("sel ppo")

    @pyqtSlot()
    def dqnSelect(self):
        ''' DQN 선택 이벤트핸들러'''
        self.selected_learn = 'dqn'
        print("sel dqn")

    @pyqtSlot()
    def changedTrading(self):
        ''' 기존에 구현된 알고리즘 트레이딩 전략 선택 이벤트핸들러'''
        self.selected_trading = [elem.text() for elem in self.tradingList.selectedItems()]

    @pyqtSlot()
    def changedSubject(self):
        ''' 모델에 필요한 지표 및 트레이딩 종목선택 이벤트핸들러'''
        self.selected_subject = [
            self.subjectList.selectedItems()[0].text(),
            [ elem.text() for elem in self.viewSubjectList.selectedItems() ],
        ]
        print(self.selected_subject)


    @pyqtSlot()
    def createModel(self):
        self.weight_path = None
        self.setInfo(file="None")

    @pyqtSlot()
    def loadModel(self):
        self.weight_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                    '', "weights files (*.h5)")[0]
        if self.weight_path!= "":
            file_name = self.weight_path.split('/')[-1]
            self.setInfo(file=file_name)
        else:
            self.weight_path = None
            self.setInfo(file="없음")


    @pyqtSlot()
    def saveModel(self):
        pass

    @pyqtSlot()
    def modelTraining(self):
        if not self.running is None and self.running.is_alive():
            return
        self.results, self.training_result = [], []
        kwargs=dict(
            mode='train',
            batch_size=32,
            initial_invest=20000,
            model_path=self.weight_path,
            selected_learn=self.selected_learn,
            selected_trading=self.selected_trading,
            selected_subject=self.selected_subject,
            ui_windows=self,
        )
        # run_program
        self.running = threading.Thread(target=program, kwargs=kwargs)
        self.running.daemon = True
        self.running.start()


    @pyqtSlot()
    def modelTest(self):
        if not self.running is None and self.running.is_alive():
            return
        self.results, self.training_result = [], []
        if self.weight_path is None:
            self.setInfo(msg="트레이딩을 하기 위해 학습된 모델을 로드해주세요.")
            return
        args_w = ['-w',self.weight_path]
        self.running = threading.Thread(target=program, args=(['-m','test','-i','10000000']+args_w, self.setInfo, self.training_result))
        self.running.daemon = True
        self.running.start()

    @pyqtSlot()
    def showGraph(self):
        plt.plot(self.training_result)
        plt.ylabel("포트폴리오 가치")
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Form()
    sys.exit(app.exec())
