# coding: utf-8

import sys
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from env.Strategies import *
from run import main as program
from pathlib import Path
import os

import threading
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("mainWindow.ui", self)

        self.weight_path = None # 현재 선택된 모델이 담긴 폴더 TODO 모델 파일? 폴더?
        self.info = ['없음','비실행중']  # ['현재 모델파일', '학습/테스트 메시지']
        self.portpolio_value_history = [] # 학습/테스트 스레드가 남길 히스토리배열
        self.setInfo("없음")
        self.running = None
        self.ui.show()

        self.tradingList.addItems([ func['name'] for func in strategies]) # 트레이딩 알고리즘 메뉴 등록

        data_list = [ elem.name for elem in Path('../daily_data').iterdir() if elem.is_dir() ]
        self.subjectList.addItems(data_list) # 종목 선택 메뉴 등록 TODO Ui에 종목 등록 유언성 필요
        self.viewSubjectList.addItems(data_list) # 지표 선택 메뉴 등록 TODO Ui에 종목 등록 유언성 필요
        self.selected_subject = [data_list[0], [data_list[0],]] # 현재 선택된 종목
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
        self.selected_trading = strategies_name_filter([elem.text() for elem in self.tradingList.selectedItems()])
        print(self.selected_trading)

    @pyqtSlot()
    def changedSubject(self):
        ''' 모델에 필요한 지표 및 트레이딩 종목선택 이벤트핸들러'''
        self.selected_subject = [
            self.subjectList.selectedItems()[0].text(),
            [ elem.text() for elem in self.viewSubjectList.selectedItems() ],
        ]
        print(self.selected_subject)


    @pyqtSlot()
    def selectModelPath(self):
        self.weight_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Model Path')
        print(self.weight_path)
        if self.weight_path!= "":
            dir_name = self.weight_path.split('/')[-1]
            self.setInfo(file=dir_name)
        else:
            self.weight_path = None
            self.setInfo(file="없음")
        print(self.episodesTextBox.toPlainText())


    @pyqtSlot()
    def modelTraining(self):
        if not self.running is None and self.running.is_alive():
            return
        self.portpolio_value_history = []

        episodes_text : str = self.episodesTextBox.toPlainText()
        if episodes_text.isnumeric():
            episodes = int(episodes_text)
        else:
            episodes = 100

        kwargs=dict(
            mode='train',
            episode=episodes,
            window_size=30,
            init_invest=100*10000,
            model_path=self.weight_path,
            addition_train=self.additionalCheckBox.isChecked(),
            selected_learn=self.selected_learn,
            selected_trading=self.selected_trading,
            selected_subject=self.selected_subject,
            ui_windows=self,
        )
        # run_program
        self.running = threading.Thread(target=program, kwargs=kwargs)
        self.running.daemon = True
        self.running.start()

        self.setInfo(msg='학습시작.')



    @pyqtSlot()
    def modelTest(self):
        if not self.running is None and self.running.is_alive():
            return
        self.portpolio_value_history = []
        if self.weight_path is None:
            self.setInfo(msg="트레이딩을 하기 위해 학습된 모델을 로드해주세요.")
            return

        episodes_text : str = self.episodesTextBox.toPlainText()
        if episodes_text.isnumeric():
            episodes = int(episodes_text)
        else:
            episodes = 100

        kwargs=dict(
            mode='test',
            episode=episodes,
            window_size=30,
            init_invest=100*10000,
            model_path=self.weight_path,
            addition_train=False,
            selected_learn=self.selected_learn,
            selected_trading=self.selected_trading,
            selected_subject=self.selected_subject,
            ui_windows=self,
        )
        # run_program
        self.running = threading.Thread(target=program, kwargs=kwargs)
        self.running.daemon = True
        self.running.start()

        self.setInfo(msg="가상트레이딩 시작.")

    @pyqtSlot()
    def showGraph(self):
        print("portfolio", self.portpolio_value_history)
        plt.plot(self.portpolio_value_history)
        plt.ylabel("포트폴리오 가치")
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Form()
    sys.exit(app.exec())
