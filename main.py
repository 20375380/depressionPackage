from PyQt5 import QtCore, QtGui
import sys
import os
from generate_sequence import seq_generator
from HMMdrawer import HMM_generator
from PyQt5.QtCore import QEventLoop, QTimer, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from Ui_ControlBoard import Ui_MainWindow

import sys


def exceptOutConfig(exctype, value, tb):
    print('My Error Information:')
    print('Type:', exctype)
    print('Value:', value)
    print('Traceback:', tb)


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))


class printThread(QThread):
    def run(self):
        for i in range(5):
            print(f"打印当前数值为：{i}.")
            self.sleep(1)
        print("End")
        print(1/0)  # 人为地引发一个异常


class ControlBoard(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ControlBoard, self).__init__()
        self.setupUi(self)
        # 下面将输出重定向到textBrowser中
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)
        self.pushButton1.clicked.connect(self.generate_seq)
        self.pushButton2.clicked.connect(self.generate_HMM)

    def generate_HMM(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self)
        if file_name:
            HMMGenerator = HMM_generator(file_name)
            HMMGenerator.generate()

    def outputWritten(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()


    def generate_seq(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)",
                                                   options=options)
        if file_name:
            print("Selected video file: ", file_name)
            video_path = file_name
            video_path = video_path.replace('/', '\\')
            print(video_path)
            file_name = os.path.basename(file_name)
            file_name = file_name.replace('.mp4', '.txt')
            file_excel = file_name.replace('.txt', '.xlsx')
            datasets_root = r'.\\picture'
            datasets_new_root = r'.\\fuse'
            txt8_path = os.path.join('.\\storage-8\\', file_name)
            excel = os.path.join('.\\excel\\', file_excel)
            seqGenerator = seq_generator(video_path, datasets_root, datasets_new_root, txt8_path, excel)  # 创建序列生成器
            seqGenerator.cut_frame()  # 裁剪视频
            seqGenerator.pretreatment()  # 融合图像
            seqGenerator.classify_main()  # 分类
            seqGenerator.last_predict()  # 分类
            picture_list = os.listdir(datasets_root)
            fuse_list = os.listdir(datasets_new_root)
            # 删除图片
            '''
            for picture_name in picture_list:
                picture_name = os.path.join(datasets_root, picture_name)  # 构造文件的完整路径
                os.remove(picture_name)  # 删除文件
            for fuse_name in fuse_list:
                fuse_name = os.path.join(datasets_new_root, fuse_name)
                os.remove(fuse_name)
            '''


if __name__ == "__main__":
    sys.excepthook = exceptOutConfig
    app = QApplication(sys.argv)
    win = ControlBoard()
    win.show()
    sys.exit(app.exec_())


